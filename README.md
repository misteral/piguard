# piguard

Fast local prompt injection detector for macOS / Linux.

Uses the [PIGuard](https://huggingface.co/leolee99/PIGuard) model (DeBERTa-v3-base fine-tune, [ACL 2025 paper](https://aclanthology.org/2025.acl-long.1468.pdf)) — converted to ONNX and served through native Rust binaries with embedded ONNX Runtime. No Python at runtime.

## Binaries

| Binary | Purpose |
|--------|---------|
| `piguard` | CLI — classify text from args, stdin, or pipe |
| `piguard-server` | Unix socket daemon — model stays in memory, ~10ms JSON responses |

The CLI automatically connects to the daemon when it's running. Without the daemon it loads the model itself (~300ms).

## Install

```bash
cargo install --path .
```

## Setup (first run)

The ONNX model needs to be exported once using the included Python script:

```bash
pip install torch transformers onnxscript
python scripts/export_onnx.py
```

This downloads the model from HuggingFace, converts it to ONNX, and saves to `~/.cache/piguard/onnx/` (~735 MB).

## CLI

```bash
piguard "Ignore previous instructions"
# 🚨 INJECTION (score: 1.000, 11.0ms)

piguard "What is the weather today?"
# ✅ BENIGN (score: 0.974, 12.9ms)

# Pipe multiple lines
echo -e "Hello world\nIgnore all instructions" | piguard

# Benchmark
piguard bench
```

## Server

Start the daemon:

```bash
piguard-server
# Model loaded in 256ms, listening on /tmp/piguard.sock
```

Query via Unix socket (newline-delimited JSON):

```bash
echo '{"text":"Ignore previous instructions"}' | nc -U /tmp/piguard.sock
# {"text":"Ignore previous instructions","label":"INJECTION","score":0.999,"latency_ms":8.3}

echo '{"text":"Hello world"}' | nc -U /tmp/piguard.sock
# {"text":"Hello world","label":"BENIGN","score":0.965,"latency_ms":9.1}
```

Options:

```bash
piguard-server --socket /tmp/custom.sock   # custom socket path
piguard-server --model-dir /path/to/onnx   # custom model directory
```

### Protocol

**Request** — one JSON object per line:
```json
{"text": "Ignore previous instructions"}
```

**Response** — one JSON object per line:
```json
{"text": "Ignore previous instructions", "label": "INJECTION", "score": 0.999, "latency_ms": 8.3}
```

Error responses:
```json
{"error": "invalid JSON: ..."}
```

### Integration with CLI

When the daemon is running, `piguard` CLI connects to it automatically — no model loading overhead:

```
# Without daemon: ~300ms total
$ time piguard "test"
real  0m0.300s

# With daemon: ~14ms total
$ time piguard "test"
real  0m0.014s
```

## Performance

| Scenario | Total time | Inference |
|----------|-----------|-----------|
| CLI (cold, no daemon) | ~300ms | ~10ms |
| CLI (via daemon) | **~14ms** | ~10ms |
| Server response | — | ~10ms |

## How it works

PIGuard is a [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) model fine-tuned for binary text classification:

- **BENIGN** — safe prompt
- **INJECTION** — prompt injection attack detected

The MOF (Mitigating Over-defense for Free) training strategy reduces false positives from trigger words like "ignore", "instructions", etc.

The Rust binaries use [ort](https://github.com/pykeio/ort) (ONNX Runtime) and [tokenizers](https://github.com/huggingface/tokenizers) — both native Rust, zero Python dependency at runtime.

## Architecture

```
                        ┌─────────────────────────┐
piguard "text" ────────►│  piguard-server          │
  (tries socket first)  │  /tmp/piguard.sock       │
                        │                           │
                        │  tokenizer ──► ONNX ──► JSON
                        │  (loaded once, ~256ms)    │
                        └─────────────────────────┘
                                    │
                           {"label":"INJECTION",
                            "score":0.999,
                            "latency_ms":8.3}
```

## Reference

```bibtex
@inproceedings{piguard2025,
  title={PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free},
  author={Hao Li and Xiaogeng Liu and Ning Zhang and Chaowei Xiao},
  booktitle={ACL},
  year={2025}
}
```

## License

MIT
