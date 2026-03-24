---
title: "I needed prompt injection detection in 14ms. Python couldn't do it."
date: 2026-03-24
status: draft
platform: linkedin
---

# I needed prompt injection detection in 14ms. Python couldn't do it.

If you're building anything with LLMs, you've probably thought about prompt injection. Someone sends "Ignore all previous instructions" and your agent happily complies.

There are models that catch this. PIGuard (ACL 2025) is a fine-tuned DeBERTa-v3-base that classifies text as BENIGN or INJECTION. It works well. The question is how you serve it.

## The Python baseline

The obvious approach: `transformers` + PyTorch.

```python
from transformers import pipeline
pipe = pipeline("text-classification", model="leolee99/PIGuard")
pipe("Ignore previous instructions")
# ~2 seconds to first result
```

Two seconds. Most of that is loading the model and initializing PyTorch. Once loaded, inference is fast — maybe 50ms per query. But if you're calling this from a CLI, from a git hook, from a shell script — you pay that 2-second tax every time.

For a library imported into a running Python process, this is fine. For a CLI tool you invoke hundreds of times, it's a dealbreaker.

## What actually takes 2 seconds

I profiled it. Roughly:

- ~800ms — importing torch and transformers
- ~400ms — loading model weights from disk
- ~500ms — initializing CUDA/CPU backend
- ~50ms — actual tokenization + inference

**95% of the time is startup overhead, not inference.**

This is the pattern with most ML-in-Python CLIs. The model is fast. Everything around it is slow.

## The Rust rewrite

I rewrote it in Rust using two crates:

- `ort` — Rust bindings for ONNX Runtime (the same runtime that powers inference in production at Microsoft, Meta, etc.)
- `tokenizers` — HuggingFace's tokenizer library, which is already written in Rust under the hood

The model stays the same — I just export it to ONNX format once with a 70-line Python script, then never touch Python again at runtime.

The core detector is straightforward:

```rust
pub struct Detector {
    session: Session,
    tokenizer: Tokenizer,
}

impl Detector {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let session = Session::builder()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .with_intra_threads(4)
            .commit_from_file(model_dir.join("model.onnx"))?;
        let tokenizer = Tokenizer::from_file(
            model_dir.join("tokenizer.json")
        )?;
        Ok(Self { session, tokenizer })
    }
}
```

Load session, load tokenizer. That's it. No framework initialization, no dynamic dispatch, no GIL.

Inference is equally minimal — tokenize, build tensors, run, softmax:

```rust
pub fn detect(&mut self, text: &str) -> Result<Detection> {
    let encoding = self.tokenizer.encode(text, true)?;
    let ids: Vec<i64> = encoding.get_ids()
        .iter().map(|&x| i64::from(x)).collect();

    let outputs = self.session.run(
        ort::inputs![
            Tensor::from_array(([1, ids.len()], ids))?,
            Tensor::from_array(([1, mask.len()], mask))?
        ]
    )?;
    // softmax → label + score
}
```

**The entire project is 530 lines of Rust.** Two binaries — a CLI and a Unix socket daemon.

## The numbers

| Scenario | Total time | Inference |
|----------|-----------|-----------|
| Python (transformers) | ~2,000ms | ~50ms |
| Rust CLI (cold start) | ~300ms | ~10ms |
| Rust CLI (via daemon) | **~14ms** | ~10ms |

Cold start went from 2 seconds to 300ms — a **6.5x improvement** just from dropping Python overhead and using ONNX Runtime directly.

But the real win is the daemon. `piguard-server` loads the model once and listens on a Unix socket. The CLI auto-detects it:

```bash
# Without daemon
$ time piguard "Ignore previous instructions"
🚨 INJECTION (score: 1.000, 11.0ms)
real  0m0.300s

# With daemon running
$ time piguard "Ignore previous instructions"
🚨 INJECTION (score: 1.000, 8.3ms)
real  0m0.014s
```

**14ms total, end to end.** That's fast enough for a git pre-commit hook. Fast enough to check every message in a chat pipeline. Fast enough that you stop thinking about it.

## The daemon trick

The daemon is dead simple — 180 lines. It accepts newline-delimited JSON over a Unix socket:

```bash
echo '{"text":"Forget everything above"}' | nc -U /tmp/piguard.sock
# {"label":"INJECTION","score":0.999,"latency_ms":8.3}
```

The CLI tries the socket first, falls back to loading the model locally. Zero configuration:

```rust
fn detect(text: &str, detector: &mut Option<Detector>) -> Result<Detection> {
    // Try daemon first
    if let Some(d) = detect_via_daemon(text) {
        return Ok(d);
    }
    // Fall back to local model
    let det = detector.get_or_insert_with(|| {
        Detector::load(&default_model_dir()).expect("failed to load model")
    });
    det.detect(text)
}
```

No service mesh. No gRPC. Just a Unix socket and JSON lines.

## The one thing Python still does

The ONNX export. You run it once:

```bash
pip install torch transformers onnxscript
python scripts/export_onnx.py
# Downloads model, exports to ~/.cache/piguard/onnx/ (~735 MB)
```

I'm not going to rewrite PyTorch model export in Rust. That's the right tool for a one-time job. After that, Python never runs again.

## The takeaway

**For CLI tools that run repeatedly, startup time is the bottleneck — not inference.** Python's import overhead alone is often 10-100x the actual computation.

The formula that worked here: train in Python, export to ONNX, serve in Rust. You keep the ML ecosystem for training and the native performance for serving. 530 lines of Rust replaced a Python dependency tree that would've been 50x heavier at runtime.

If you're building developer tools that wrap ML models — consider whether your users are paying a 2-second tax on every invocation.

The project is open source: [github.com/piguard](https://github.com/piguard) (MIT)

#PromptInjection #Rust #LLMSecurity #ONNX #DeveloperTools
