# websam-export

Export pipeline that turns the SAM3 tracker-video and EdgeTAM checkpoints into
the ONNX graph set + manifest consumed by the WebSAM browser runtime. Python
3.12+, managed with [uv](https://docs.astral.sh/uv/). Not part of the pnpm
workspace.

At M0 this package ships the **executable spec** (`websam_export.spec`), the
**manifest builder/validator** (`websam_export.manifest`), and their tests.
The export stages themselves land in later milestones, but they must conform
to the contract encoded here — the browser runtime compiles against these
exact shapes.

## Setup

```sh
uv sync                 # light: spec + manifest + onnx/ort tooling
uv sync --extra export  # heavy: torch + transformers + onnxslim (pipeline runs)
uv run --group dev pytest -q
```

## Pipeline overview

```
wrappers -> dynamo export -> onnxslim -> quant ladder -> parity vs goldens -> manifest -> upload
```

1. **Wrappers** — thin `torch.nn.Module` wrappers around the HF modules
   (`modeling_sam3_tracker_video.py`) that cut the model at the four graph
   boundaries in `spec.py` (`image_encoder`, `memory_attention`,
   `mask_decoder`, `memory_encoder`) and freeze the memory-attention KV
   length to the tier maximum (validity driven by `memory_mask`).
2. **Dynamo export** — `torch.onnx.export(dynamo=True)` per wrapper at the
   tier's opset; dynamic axes only where `spec.py` declares a symbolic dim
   (e.g. `num_points`).
3. **onnxslim** — graph simplification/constant folding; the slimmed graph
   must keep the exact input/output names and dims from the spec.
4. **Quant ladder** — fp32 → fp16 → (selective) int8; each rung re-runs
   parity before it is allowed to ship.
5. **Parity vs committed goldens** — every graph runs under onnxruntime
   (pinned to the 1.27 line, matching the runtime's `onnxruntime-web`
   `>=1.27.0 <1.28.0`) against golden input/output tensors committed to the
   repo; per-rung tolerances gate promotion.
6. **Manifest** — `build_manifest(spec, files)` records schemaVersion 1,
   tier, opset, toolchain versions, typed graph signatures, and a streamed
   SHA-256 per artifact; `validate_manifest` is the final gate before upload.
7. **Upload** — artifacts + manifest are pushed to the distribution host
   (see distribution policy below).

## Spike ladder (S0–S6)

Each spike has a go/no-go bar; a red bar stops the ladder and forces a
contract or approach change before more work is stacked on top.

| Spike | Question it retires | Go bar | No-go response |
| --- | --- | --- | --- |
| **S0** — toy dynamo export | Does `torch.onnx.export(dynamo=True)` + onnxslim + ORT run at all on a toy module with our dtypes? | Toy graph loads and runs in ORT; max abs err < 1e-6 vs eager | Fall back to TorchScript-path export |
| **S1** — image encoder | Does the SAM3 vision backbone export cleanly at 1008/560? | fp32 parity max abs err < 1e-4 on 8 golden frames | Split backbone/neck into separate graphs |
| **S2** — mask decoder | Do prompt encoding + two-way transformer export with dynamic `num_points`? | Mask IoU vs eager > 0.99 on golden prompts; dynamic axis honored | Freeze `num_points` per prompt-count bucket |
| **S3** — memory attention, fixed KV | Does frozen-KV (51904 / 16064 / 1856) + boolean mask reproduce eager memory attention? | Conditioned-feature max abs err < 1e-3; masked padding provably inert | Re-derive bank constants; revisit mask dtype |
| **S4** — streaming memory bank | Does the full per-frame loop (encode → attend → decode → memory-encode → evict) match eager over a real clip? | Per-frame mask IoU > 0.98 across a 100-frame clip; tpos rule verified (cond→6, offset k→k−1) | Fix bank update order / tpos before quant |
| **S5** — quant ladder | How far down the ladder before quality breaks? | fp16: IoU drop < 0.005; int8: IoU drop < 0.02 on the eval clip set | Ship the last green rung only |
| **S6** — in-browser perf | Does the tier hit its latency budget under onnxruntime-web (wasm/webgpu)? | SAM3_560 interactive on mid-tier laptop webgpu; EdgeTAM realtime-ish; no >2x memory blowup vs native | Descope tier or move rung down the ladder |

## Distribution policy

* **EdgeTAM** — Apache-2.0 licensed; exported artifacts are **re-hosted** by
  us (Hugging Face Hub) and downloaded directly by the browser runtime with
  manifest SHA-256 verification.
* **SAM3** — gated license; we do **not** re-host weights or exported graphs
  by default. Users run this pipeline themselves against their own gated
  download (`huggingface-hub` auth), producing local artifacts + manifest
  that the runtime loads from their own storage/origin.

## Layout

```
pyproject.toml
src/websam_export/
  spec.py       # ExportSpec/GraphSpec/TensorSpec, memory-bank constants, TIERS, tpos rule
  manifest.py   # build_manifest / validate_manifest / sha256_file
tests/
  test_spec.py      # KV-length arithmetic, tpos mapping, spec self-validation
  test_manifest.py  # manifest round-trip, tamper rejection, streamed sha256
```
