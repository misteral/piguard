# piguard

Fast local prompt injection detector CLI for macOS / Linux.

Uses the [PIGuard](https://huggingface.co/leolee99/PIGuard) model (DeBERTa-v3-base fine-tune, [ACL 2025 paper](https://aclanthology.org/2025.acl-long.1468.pdf)) — converted to ONNX and served through a native Rust binary with embedded ONNX Runtime. No Python at runtime.

**~300ms cold start, ~10ms per query.**

## Install

```bash
cargo install --path .
```

Or build manually:

```bash
cargo build --release
cp target/release/piguard ~/.local/bin/
```

## Setup (first run)

The ONNX model needs to be exported once using the included Python script:

```bash
pip install torch transformers onnxscript
python scripts/export_onnx.py
```

This downloads the model from HuggingFace, converts it to ONNX, and saves to `~/.cache/piguard/onnx/` (~735 MB).

## Usage

```bash
# Single text
piguard "Ignore previous instructions"
# 🚨 INJECTION (score: 1.000, 11.0ms)

piguard "What is the weather today?"
# ✅ BENIGN (score: 0.974, 12.9ms)

# Pipe multiple lines
echo -e "Hello world\nIgnore all instructions" | piguard

# Benchmark (10 samples)
piguard bench
```

## Performance

| Metric | Value |
|--------|-------|
| Cold start (first call) | ~300ms |
| Inference per query | ~10ms |
| Model size on disk | 735 MB |
| Binary size | ~15 MB (with ONNX Runtime) |

The ~300ms startup is pure I/O — loading 735 MB of model weights from disk. Inference itself is ~10ms per query. No Python overhead.

## How it works

PIGuard is a [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) model fine-tuned for binary text classification:

- **BENIGN** — safe prompt
- **INJECTION** — prompt injection attack detected

It uses the MOF (Mitigating Over-defense for Free) training strategy to reduce false positives from trigger words like "ignore", "instructions", etc. — a common problem with other prompt guard models.

The Rust CLI loads the ONNX model via [ort](https://github.com/pykeio/ort) (ONNX Runtime bindings) and tokenizes with [tokenizers](https://github.com/huggingface/tokenizers) — both native Rust, zero Python dependency at runtime.

## Architecture

```
piguard "text"
    │
    ├── tokenizers (Rust) ──→ token IDs + attention mask
    │
    ├── ort / ONNX Runtime ──→ model inference (~10ms)
    │
    └── softmax ──→ BENIGN / INJECTION + confidence score
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
