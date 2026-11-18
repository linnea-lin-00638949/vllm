# Kimi first-boundary optimization: five-version comparison with an unmodified baseline

This report accompanies the integration into `linnea-lin-00638949/vllm:improve/pr68112-linear-boundary`. The singleton-delta contribution and its commit history are retained. The integration adds the earlier local Kimi fast lookup and bulk slicing while preserving the source branch's bounded integer indexing.

## Versions

| Result key | Version |
|---|---|
| unfixed | Unmodified pre-fix baseline, `c26c9bfeb7ee2c804fe55b425afd049809814582` |
| base | Initial A (already fixed), `cbc5b29ec71c645855551a133e0d22a96c2c35a4` |
| before | Initial A + [archived local optimization patch](pre-singleton-optimization.patch): Kimi first-close lookup and bulk slices, still full-prefix fallback |
| source | QwertyJack singleton-delta version, `d213a3bc738127032524a639bbb834644022af73` |
| after | Integration production code at `dbbe19111bd9bf693fda94dd3d12796569956008` (unchanged by this report update): Kimi fast lookup + singleton fallback + bulk slices + bounded integer indexing |

The pre-singleton local version did not have a standalone commit; the exact production patch is included rather than presenting it as a released revision. [Five-version raw results](performance-five-version-results.json) include every timing sample and final production source SHA256 values, so the measured code can be checked independently of subsequent commits.

## Method

macOS ARM64, Python 3.12.14, torch 2.14.0, XGrammar 0.2.7. GPT-2 tokenizer/config only; no weights or accelerator. Actual manager, parser, TokenIdsView and Request classes are used. Archived complete source modules provide the four reference arms; the candidate imports from the current checkout. No extracted AST or fake parser.

Same process, one Torch thread, 100 warmups per arm/case, nine alternating-order rounds, 300 calls per arm per round, GC disabled while timing. The four fixed arms assert the same correct boundary before measurement. The unfixed arm separately asserts its historical behavior; incorrect outputs remain measured and are explicitly marked rather than silently treated as equivalent work. Tests and timing runs were not intentionally overlapped. Figures are microseconds per `_get_constraint_bounds` call, medians of nine rounds; raw min/max/samples are retained. Context sizes test this helper, not the GPT-2 model context limit.

## Results

All five arms below were rerun together. The new baseline column is not spliced into earlier measurements. Units: microseconds per boundary lookup.

| History tokens | Shape | Draft tokens | Unmodified | Initial A | Previous local A | QwertyJack version | Final integration |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | think.k=1 | 1 | 4.74 | 4.74 | 1.14 | 4.91 | 1.20 |
| 128 | think.k=7 | 7 | 4.89 | 16.60 | 1.15 | 27.34 | 1.15 |
| 128 | think.k=16 | 16 | 5.13 | 40.17 | 1.33 | 54.28 | 1.45 |
| 128 | think.k=32 | 32 | 5.26 | 76.90 | 1.55 | 88.97 | 1.52 |
| 128 | close_last | 13 | 7.51 | 35.14 | 1.86 | 49.28 | 1.87 |
| 128 | split_close_last | 1 | 4.61 | 4.64 | 3.72 | 4.71 | 3.69 |
| 128 | close_reopen | 25 | 4.75† | 70.81 | 1.86 | 90.37 | 1.85 |
| 128 | split_close_reopen.pad=0 | 13 | 4.65† | 39.31 | 43.62 | 49.96 | 36.49 |
| 128 | split_close_reopen.pad=16 | 29 | 4.85† | 83.77 | 93.64 | 91.84 | 66.11 |
| 128 | split_close_reopen.pad=64 | 77 | 6.00† | 240.95 | 270.31 | 204.20 | 145.91 |
| 128 | json | 7 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |
| 32768 | think.k=1 | 1 | 362.03 | 363.85 | 0.99 | 4.43 | 0.97 |
| 32768 | think.k=7 | 7 | 362.78 | 373.02 | 1.08 | 26.69 | 1.06 |
| 32768 | think.k=16 | 16 | 366.90 | 401.37 | 1.23 | 52.42 | 1.23 |
| 32768 | think.k=32 | 32 | 365.91 | 445.38 | 1.47 | 93.76 | 1.49 |
| 32768 | close_last | 13 | 364.74 | 391.64 | 1.82 | 50.78 | 1.79 |
| 32768 | split_close_last | 1 | 363.81 | 363.33 | 3.75 | 4.84 | 3.81 |
| 32768 | close_reopen | 25 | 361.34† | 429.91 | 1.86 | 95.20 | 1.82 |
| 32768 | split_close_reopen.pad=0 | 13 | 362.58† | 404.41 | 44.90 | 51.94 | 36.55 |
| 32768 | split_close_reopen.pad=16 | 29 | 361.33† | 444.79 | 91.40 | 93.92 | 64.30 |
| 32768 | split_close_reopen.pad=64 | 77 | 362.08† | 604.15 | 272.50 | 219.80 | 148.69 |
| 32768 | json | 7 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |

**† Incorrect boundary:** the unfixed version lets the later reopen erase the earlier close. Its elapsed time is recorded as a historical baseline, not a valid performance win. The old version does less constraint checking and returns the wrong result in these cases. For either history length:

| Shape | Unmodified boundary | Correct boundary in all four fixed versions |
|---|---:|---:|
| close_reopen | 26 | 13 |
| split_close_reopen.pad=0 | 14 | 1 |
| split_close_reopen.pad=16 | 30 | 1 |
| split_close_reopen.pad=64 | 78 | 1 |

`close_last` and `split_close_last` do not reopen thinking: all five versions produce the correct boundary, providing equivalent-result comparisons in addition to normal thinking and already-in-JSON cases. For example, at 32K history, normal thinking with draft=7 takes 362.78 μs unmodified versus 1.06 μs integrated; a complete close without reopening takes 364.74 versus 1.79 μs. These are CPU helper measurements, not serving throughput multipliers.

`split_close_reopen.pad=N` completes a close marker split across steps, then contains N plain tokens before reopening. N=16/64 are synthetic performance extensions, not additional natural incidents. The tokenizer expands the protocol markers; these draft lengths are fixture lengths, not production Kimi token IDs.

In this five-arm run at 32K history, the final cross-step cases reduce time versus the previous local optimization by approximately 19%, 30%, and 45% for draft lengths 13, 29, and 77. Normal thinking and complete-marker cases retain the previous local fast path. Small sub-microsecond differences, especially the already-in-JSON early exit, are not evidence of meaningful speedups or regressions.

For fixed Kimi marker lengths, both the direct lookup and singleton fallback are linear in draft length, with bounded history reads. This is not a universal complexity guarantee for parsers that scan/decode the full history. No serving throughput, GPU/NPU latency, model accuracy, or real rejection-sampler improvement is established by these measurements.

Earlier [four-version samples](performance-results.json) and [their independent reproduction run](performance-reproduction-results.json) remain archived separately. Their numbers are not mixed into this table. The zero-context archived patch reconstructs exactly the original pre-singleton source files.

## Correctness and checks

This addition changes only the benchmark/report/data. Production source hashes match the previously validated integration. The new benchmark ran 22 shapes/history combinations across five arms: all 88 fixed-arm outputs are correct, the unfixed baseline is correct in 14 and has the expected wrong boundary in eight. The checks below are the preceding integration results, not a new full-suite run for this reporting-only change.

- Final-tree targeted suite: **107 passed**. Includes every relevant Kimi fixture split/prefix, committed/uncommitted drafts, enabled/disabled thinking, atomic and unequal marker tokenizations, bounded view slices, and the new shared-parser tests.
- Expanded existing suite: **335 passed** (overlaps the targeted suite; do not add these counts). Request, Kimi, BaseThinking, GPT-OSS, MiniMax M3, parser engine and response-template tests.
- Four-way real-manager differential: **4,184 boundary outputs across ten parser variants are exactly identical in all four arms**. DeepSeek R1/V3, HY V3, Step3, Cohere Command3/4, MiniMax M2/M3, K2 Horizon and Kimi K3. Fixed cases plus seeded random marker sequences; finite coverage, not proof for all plugins/configurations.
- Fast-lookup negative control on the source branch's original parser: **2 assertion failures / 1 pass**, then all three pass on the integration. The failures are absent first-close lookup results for GPT-2/atomic markers, not setup errors.
- Ruff 0.17.2 check/format and `git diff --check` pass for the changed production/test files. Full pre-commit and remote CI are not claimed.
- Local ARM CPU rejects speculative decoding without Triton during configuration, so prospective GPU mask/rejection-sampler and full serving tests were not rerun. The source PR's separately reported Linux CPU grammar-mask results do not validate this integration's current head.

## Reproduction

Use a compatible vLLM development environment with the cached GPT-2 tokenizer/config and the three historical commits above. No model weights are needed. From this checkout:

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

## Subsequent generation validation

The separate [2026-11-18 generation report](generation-regression.md) adds fixed-draft and multi-turn CPU checks, native deterministic before/after results, 100 normal HTTP replays and 10 multi-turn HTTP requests. Accelerator evidence uses a downstream API adapter and direct decode; full PD routing was not validated. This does not change the scope of the CPU timings above.
