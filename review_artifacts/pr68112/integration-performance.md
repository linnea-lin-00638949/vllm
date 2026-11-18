# Kimi first-boundary optimization: four-version comparison

This report accompanies the integration into `QwertyJack/vllm:improve/pr68112-linear-boundary`. The singleton-delta contribution and its commit history are retained. The integration adds the earlier local Kimi fast lookup and bulk slicing while preserving the source branch's bounded integer indexing.

## Versions

| Result key | Version |
|---|---|
| base | Initial A, `cbc5b29ec71c645855551a133e0d22a96c2c35a4` |
| before | Initial A + [archived local optimization patch](pre-singleton-optimization.patch): Kimi first-close lookup and bulk slices, still full-prefix fallback |
| source | QwertyJack singleton-delta version, `d213a3bc738127032524a639bbb834644022af73` |
| after | Integration in this commit: Kimi fast lookup + singleton fallback + bulk slices + bounded integer indexing |

The pre-singleton local version did not have a standalone commit; the exact production patch is included rather than presenting it as a released revision. [Raw results](performance-results.json) include every timing sample and final production source SHA256 values, so the measured code can be checked independently of subsequent commits.

## Method

macOS ARM64, Python 3.12.14, torch 2.14.0, XGrammar 0.2.7. GPT-2 tokenizer/config only; no weights or accelerator. Actual manager, parser, TokenIdsView and Request classes are used. Archived complete source modules provide the three reference arms; the candidate imports from the current checkout. No extracted AST or fake parser.

Same process, one Torch thread, 100 warmups per arm/case, nine alternating-order rounds, 300 calls per arm per round, GC disabled while timing. All arms assert the same expected boundary before measurement. Tests and timing runs were not intentionally overlapped. Figures are microseconds per `_get_constraint_bounds` call, medians of nine rounds; raw min/max/samples are retained. Context sizes test this helper, not the GPT-2 model context limit.

## Results

| History tokens | Shape | Draft tokens | Initial A | Previous local A | QwertyJack version | Final integration |
|---:|---|---:|---:|---:|---:|---:|
| 128 | think.k=1 | 1 | 4.38 | 1.05 | 4.50 | 1.03 |
| 128 | think.k=7 | 7 | 16.13 | 1.16 | 26.83 | 1.14 |
| 128 | think.k=16 | 16 | 37.56 | 1.30 | 50.73 | 1.27 |
| 128 | think.k=32 | 32 | 77.62 | 1.54 | 91.35 | 1.53 |
| 128 | close_reopen | 25 | 72.27 | 1.94 | 93.07 | 1.94 |
| 128 | split_close_reopen.pad=0 | 13 | 40.47 | 45.26 | 52.21 | 38.61 |
| 128 | split_close_reopen.pad=16 | 29 | 85.68 | 97.94 | 94.72 | 69.10 |
| 128 | split_close_reopen.pad=64 | 77 | 256.20 | 287.15 | 216.81 | 155.85 |
| 128 | json | 7 | 0.07 | 0.07 | 0.07 | 0.07 |
| 32768 | think.k=1 | 1 | 411.48 | 1.08 | 4.89 | 1.03 |
| 32768 | think.k=7 | 7 | 390.78 | 1.16 | 27.94 | 1.14 |
| 32768 | think.k=16 | 16 | 412.59 | 1.32 | 54.23 | 1.32 |
| 32768 | think.k=32 | 32 | 455.57 | 1.55 | 98.79 | 1.54 |
| 32768 | close_reopen | 25 | 447.76 | 1.96 | 99.69 | 1.89 |
| 32768 | split_close_reopen.pad=0 | 13 | 414.64 | 46.13 | 54.68 | 38.14 |
| 32768 | split_close_reopen.pad=16 | 29 | 468.50 | 98.13 | 101.95 | 70.57 |
| 32768 | split_close_reopen.pad=64 | 77 | 652.06 | 294.58 | 238.29 | 161.30 |
| 32768 | json | 7 | 0.07 | 0.14 | 0.08 | 0.10 |

`split_close_reopen.pad=N` completes a close marker split across steps, then contains N plain tokens before reopening. N=16/64 are synthetic performance extensions, not additional natural incidents. The tokenizer expands the protocol markers; these draft lengths are fixture lengths, not production Kimi token IDs.

At 32K history, the final cross-step cases reduce time versus the previous local optimization by approximately 17%, 28%, and 45% for draft lengths 13, 29, and 77. Relative to the QwertyJack version they improve by approximately 30%, 31%, and 32%. Normal thinking and complete-marker cases retain the previous local fast path. Small sub-microsecond differences (especially the already-in-JSON early exit) are not evidence of a meaningful speedup or regression.

For fixed Kimi marker lengths, both the direct lookup and singleton fallback are linear in draft length, with bounded history reads. This is not a universal complexity guarantee for parsers that scan/decode the full history. No serving throughput, GPU/NPU latency, model accuracy, or real rejection-sampler improvement is established by these measurements.

The checked-in reproduction script was also executed successfully; [its independent timing samples](performance-reproduction-results.json) are retained separately rather than mixed into the table. The zero-context archived patch was verified to reconstruct exactly the original pre-singleton source files.

## Correctness and checks

- Final-tree targeted suite: **107 passed**. Includes every relevant Kimi fixture split/prefix, committed/uncommitted drafts, enabled/disabled thinking, atomic and unequal marker tokenizations, bounded view slices, and the new shared-parser tests.
- Expanded existing suite: **335 passed** (overlaps the targeted suite; do not add these counts). Request, Kimi, BaseThinking, GPT-OSS, MiniMax M3, parser engine and response-template tests.
- Four-way real-manager differential: **4,184 boundary outputs across ten parser variants are exactly identical in all four arms**. DeepSeek R1/V3, HY V3, Step3, Cohere Command3/4, MiniMax M2/M3, K2 Horizon and Kimi K3. Fixed cases plus seeded random marker sequences; finite coverage, not proof for all plugins/configurations.
- Fast-lookup negative control on the source branch's original parser: **2 assertion failures / 1 pass**, then all three pass on the integration. The failures are absent first-close lookup results for GPT-2/atomic markers, not setup errors.
- Ruff 0.17.2 check/format and `git diff --check` pass for the changed production/test files. Full pre-commit and remote CI are not claimed.
- Local ARM CPU rejects speculative decoding without Triton during configuration, so prospective GPU mask/rejection-sampler and full serving tests were not rerun. The source PR's separately reported Linux CPU grammar-mask results do not validate this integration's current head.

## Reproduction

Use a compatible vLLM development environment with the cached GPT-2 tokenizer/config and the two historical commits above. No model weights are needed. From this checkout:

```bash
VLLM_TARGET_DEVICE=cpu PYTHONPATH=. HF_HUB_OFFLINE=1 \
  .venv/bin/python review_artifacts/pr68112/benchmark.py /tmp/boundary-benchmark.json

VLLM_TARGET_DEVICE=cpu PYTHONPATH=. HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest tests/v1/structured_output/test_structured_output_manager.py \
  tests/v1/test_request.py tests/reasoning/test_kimi_k3_reasoning_parser.py \
  -k 'not mask and (kimi or single_token_checks or token_ids_view or clear_backend_without_grammar)' -q

VLLM_TARGET_DEVICE=cpu PYTHONPATH=. HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest tests/v1/test_request.py \
  tests/reasoning/test_kimi_k3_reasoning_parser.py \
  tests/reasoning/test_base_thinking_reasoning_parser.py \
  tests/reasoning/test_gptoss_reasoning_parser.py \
  tests/reasoning/test_minimax_m3_reasoning_parser.py \
  tests/parser/engine/test_parser_engine.py tests/parser/test_response_template.py -q
```

The benchmark refuses to overwrite an existing result file. Environment versions and timing noise can change absolute measurements; compare all arms on the same machine. AI assistance was used to integrate the code, run these checks and prepare this report.
