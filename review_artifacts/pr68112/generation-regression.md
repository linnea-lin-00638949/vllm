# Final A: deterministic and HTTP generation regression (2026-11-18)

The final combined production code is `dbbe19111bd9bf693fda94dd3d12796569956008`; the report-only parent is `a68eff5e54cd595a76a6c9a1936bd54ba106f2be`. This supplements the [five-version CPU microbenchmark](integration-performance.md), which is not a serving-throughput measurement.

## Evidence and scope

| Check | Result | What actually ran |
|---|---|---|
| Fixed-draft CPU boundary reproducer | Old 4/4 exposed the bug; final 4/4 passed | Actual upstream source, manager/parser/XGrammar; GPT-2 tokenizer, no model weights |
| Historical-turn CPU checks | 48/48 boundaries; 24/24 JSON grammar commits | Actual upstream Request/manager; 0, 1, 4, 32 prior think blocks; close marker splits; prospective and committed paths |
| Native deterministic replay | Old incident 3/3 FSM rejections; final 3/3 accepted; 6/6 controls passed in both | Downstream adapter; real Kimi tokenizer/XGrammar and unchanged Ascend rejection/residual kernels; **constructed logits**, three seeds |
| Observer equivalence | 9/9 traced results equal untraced final results | Same deterministic fixture, excluding only the variant label |
| Normal HTTP replay | 100/100 HTTP 200, strict empty JSON object, stop | Real Kimi target weights and DSpark proposals; **decode endpoint directly**, no injection |
| Normal HTTP multi-turn | 10/10 HTTP 200, strict empty JSON object, stop | 1 or 4 completed historical think turns, five requests each; direct decode |
| Real close/reopen observation | 15/100 requests, 15 steps | All 15 reached JSON sampling; reopen masked; all committed blocks accepted |
| Full prefill/decode disaggregation | **Not validated** | Routing/registration failed; first full-path request timed out after about 600 s without an FSM event |

The accelerator runs used an API adapter on downstream vLLM `a200d3cf4a4d3fc101aab18bba911c919a1ce87e` and vllm-ascend `99432e0fafc10c2ea5a40ca01d6048895bc691a0` (vLLM 0.30.1.dev30), not current upstream main binaries or the newly submitted Ascend patch packaging. The old manager returns an integer boundary rather than `ConstraintBounds`; the Kimi finder likewise returns an integer offset. Seven transplanted helper functions/classes matched the final source by AST comparison. This is evidence for the adapted algorithm, not proof that every new integration/version path works on hardware.

The isolated environment allocated eight 16-device Ascend A3 hosts (four P, four D). Successful requests went directly to a D engine (TP16, DP4, DCP16, PP1), which processed the prompt itself. There was **no P-to-D KV handoff**, and the response's `kv_transfer_params` was null. The full path would be client → router → prefill → KV transfer → decode → response; that path did not pass this run. All eight runtime code copies were hash checked. The isolated resources were released after collection.

## Recorded deterministic failure

For the same seven-token draft `[13, 163588, 39964, 163589, 163587, 39964, 163589]` (`. <|close|>think<|sep|><|open|>think<|sep|>`), all three seeds had identical pre-mask inputs across arms:

| Arm | Boundary | Reopen allowed | Committed IDs | FSM |
|---|---:|---:|---|---|
| Unfixed | 8 | 1 | `[13,163588,39964,163589,163587,12092]` | Reject |
| Final adapter | 4 | 0 | `[13,163588,39964,163589,13071]` | Accept |

The fixture uses a thin harness binding production methods, not a full scheduler/model-runner execution. Logits are constructed to make acceptance deterministic; they are **not captured model logits**. See [all three arms and controls](generation/deterministic-results.json). No changes to the Ascend sampler, acceptance cap or residual kernel are included in A.

## Real HTTP trigger

One of the 15 observed steps had 203 prior tokens and raw proposal:

```text
[163588,39964,163589,163587,39964,163589,13071]
 <close> think <sep> <open> think <sep>  ...
```

The adapter returned boundary 3. Grammar validation scheduled `[163588,39964,163589,-1,-1,-1,-1]`. The reopen mask was zero at row 3; later ones in `[1,1,1,0,1,1,1,1]` belong to invalid padding rows and do not revoke row 3. The actual committed block was `[163588,39964,163589,90]`, accepted by the FSM, followed by a valid `{}` response. [Sanitized observations](generation/http-summary.json) preserve all 15 steps. These host traces establish boundary/mask/commit behavior; they do not reconstruct every device probability or random threshold.

## Replaying the HTTP request

The [exact payload](generation/http-33.request.json) has SHA256 `f57d11b88c1a051cbbcca0cb04e9ecc1cdabfc19d0f592da4a69fb244ccce4d3`. “33 tokens” means the partial thinking prefix, not the complete prompt (202 Kimi tokens). It uses seed 5340304686692337680, temperature/top-p 1, top-k 0, no penalties, a strict empty-object schema, and `/v1/completions`. Use a compatible Kimi tokenizer/model and DSpark deployment; changing hardware, weights or scheduling need not reproduce identical drafts even with the same seed.

```bash
curl --fail-with-body "$BASE_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @review_artifacts/pr68112/generation/http-33.request.json
```

[100 per-request results](generation/http-100-results.json) and [10 multi-turn results](generation/http-multiturn-results.json) retain response text, JSON content, finish reason, token counts and KV-transfer field. [CPU multi-turn results](generation/cpu-multiturn-results.json) record all 48 combinations; `grammar_and_json_commit: false` denotes the 24 committed-boundary-only cases, not failed assertions. The four-case [fixed](generation/final-check.json) and [unfixed](generation/baseline-check.json) CPU outputs are included too. Private endpoint/host paths and raw infrastructure logs are omitted.

An earlier old-code HTTP run had 82/100 FSM failures, and an earlier minimal-A run passed 100/100. Those used a different entry environment; they are historical context, **not a controlled same-runtime HTTP A/B against this final run**. In particular, 15/100 is draft incidence, not a failure rate comparable to 82/100. This experiment establishes schema correctness on this regression workload, not broad model accuracy, full PD correctness or end-to-end speedup.
