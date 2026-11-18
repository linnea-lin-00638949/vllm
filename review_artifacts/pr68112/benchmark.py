# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare five boundary implementations, including the unfixed CPU baseline."""

import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

import vllm
from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("boundary-benchmark.json")
if OUTPUT.exists():
    raise SystemExit(f"Refusing to overwrite {OUTPUT}")
assert Path(vllm.__file__).resolve().is_relative_to(ROOT), vllm.__file__
TEMP = tempfile.TemporaryDirectory(prefix="boundary-benchmark-")
SNAPSHOTS = Path(TEMP.name)
PATHS = {
    "manager": "vllm/v1/structured_output/__init__.py",
    "view": "vllm/v1/token_buffer.py",
    "kimi": "vllm/reasoning/kimi_k3_reasoning_parser.py",
}
for arm, rev in (
    ("unfixed", "c26c9bfeb7ee2c804fe55b425afd049809814582"),
    ("base", "cbc5b29ec71c645855551a133e0d22a96c2c35a4"),
    ("source", "d213a3bc738127032524a639bbb834644022af73"),
):
    for name, path in PATHS.items():
        content = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=ROOT)
        (SNAPSHOTS / f"{arm}_{name}.py").write_bytes(content)
        if arm == "base":
            destination = SNAPSHOTS / "before-tree" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
subprocess.run(
    ["git", "apply", "--unidiff-zero", str(HERE / "pre-singleton-optimization.patch")],
    cwd=SNAPSHOTS / "before-tree",
    check=True,
)
for name, path in PATHS.items():
    (SNAPSHOTS / f"before_{name}.py").write_bytes(
        (SNAPSHOTS / "before-tree" / path).read_bytes()
    )


def load(name):
    spec = importlib.util.spec_from_file_location(name, SNAPSHOTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


torch.set_num_threads(1)
config = VllmConfig(
    model_config=ModelConfig(model="gpt2"),
    device_config=DeviceConfig(device="cpu"),
    structured_outputs_config=StructuredOutputsConfig(
        backend="xgrammar", reasoning_parser="kimi_k3"
    ),
)
managers = {}
for label in ("unfixed", "base", "before", "source", "after"):
    if label == "after":
        managers[label] = StructuredOutputManager(config)
        continue
    view = load(label + "_view")
    kimi = load(label + "_kimi")
    module = load(label + "_manager")
    module.TokenIdsView = view.TokenIdsView
    manager = module.StructuredOutputManager(config)
    manager.reasoner_cls = kimi.KimiK3ReasoningParser
    managers[label] = manager
tok = managers["after"].tokenizer
word = tok.encode(" hello")[0]
close = tok.encode("<|close|>think<|sep|>")
reopen = tok.encode("<|open|>think<|sep|>")
params = SamplingParams(
    structured_outputs=StructuredOutputsParams(json='{"type":"object"}')
)
results = []
for context in (128, 32768):
    for case, extra, draft, ended in (
        *[(f"think.k={k}", [], [word] * k, False) for k in (1, 7, 16, 32)],
        ("close_last", [], [word] + close, False),
        ("split_close_last", close[:-1], close[-1:], False),
        ("close_reopen", [], [word] + close + reopen, False),
        *[
            (
                f"split_close_reopen.pad={n}",
                close[:-1],
                close[-1:] + [word] * n + reopen,
                False,
            )
            for n in (0, 16, 64)
        ],
        ("json", [], [word] * 7, True),
    ):
        requests = {
            name: Request(
                request_id=name,
                prompt_token_ids=[word] * context + reopen + extra,
                sampling_params=params,
                pooling_params=None,
                reasoning_ended=ended,
            )
            for name in managers
        }
        expected = (
            0
            if ended
            else (
                len(draft) + 1
                if case.startswith("think")
                else 1
                if extra
                else len(close) + 1
            )
        )
        boundaries = {}
        correct = {}
        for name, manager in managers.items():
            bounds = manager._get_constraint_bounds(requests[name], draft)
            boundaries[name] = tuple(bounds)
            correct[name] = tuple(bounds) == (expected, expected)
            # The original code skips the constraint when the full draft reopens
            # thinking. Measure it, but never treat the wrong answer as a win.
            historical = len(draft) + 1 if "reopen" in case else expected
            want = historical if name == "unfixed" else expected
            assert tuple(bounds) == (want, want), (name, case, bounds, want)
            for _ in range(100):
                manager._get_constraint_bounds(requests[name], draft)
        gc.collect()
        gc.disable()
        values = {name: [] for name in managers}
        try:
            for rep in range(9):
                for name in (
                    list(managers) if rep % 2 == 0 else list(reversed(managers))
                ):
                    manager, req = managers[name], requests[name]
                    start = time.perf_counter_ns()
                    for _ in range(300):
                        manager._get_constraint_bounds(req, draft)
                    values[name].append((time.perf_counter_ns() - start) / 300 / 1000)
        finally:
            gc.enable()
        row = {
            "case": case,
            "context": context,
            "draft_len": len(draft),
            "expected_boundary": expected,
            "observed_bounds": boundaries,
            "correct_boundary": correct,
            "timing": {
                n: {
                    "median_us": statistics.median(v),
                    "min_us": min(v),
                    "max_us": max(v),
                    "samples_us": v,
                }
                for n, v in values.items()
            },
        }
        results.append(row)
        print(json.dumps(row), flush=True)
for manager in managers.values():
    manager.clear_backend()
OUTPUT.write_text(
    json.dumps(
        {
            "results": results,
            "versions": {
                "unfixed": "c26c9bfeb7ee2c804fe55b425afd049809814582",
                "base": "cbc5b29ec71c645855551a133e0d22a96c2c35a4",
                "before": "base + pre-singleton-optimization.patch",
                "source": "d213a3bc738127032524a639bbb834644022af73",
                "after": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
            },
            "source_sha256": {
                arm: {
                    name: hashlib.sha256(
                        (
                            ROOT / path
                            if arm == "after"
                            else SNAPSHOTS / f"{arm}_{name}.py"
                        ).read_bytes()
                    ).hexdigest()
                    for name, path in PATHS.items()
                }
                for arm in managers
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
            },
            "method": (
                "same process; intact production classes; 9 alternating rounds, "
                "300 calls each, 100 warmups, GC disabled"
            ),
        },
        indent=2,
    )
    + "\n"
)

TEMP.cleanup()
