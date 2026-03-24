# PiGuard — Methodology

> Distilled on 2026-03-24. 12 atoms extracted.

## Problem Statement

Need to detect prompt injection attacks locally, fast, from CLI. Existing options:
- **Python transformers pipeline** — works but 2s startup per invocation, unusable for scripting/hooks
- **Cloud APIs** — latency, cost, privacy concerns for security-sensitive text
- **Regex/heuristics** — fast but brittle, high false positive rate on words like "ignore"

Constraints: must be local (no cloud), must be fast enough for git hooks (<50ms), must have low false positive rate.

## Approach

**"Train in Python, serve in Rust"** — use HuggingFace ecosystem for model export, Rust + ONNX Runtime for inference.

1. Take an academic model (PIGuard, DeBERTa-v3-base, ACL 2025) — don't train your own
2. Export to ONNX once via Python script
3. Build Rust CLI using `ort` + `tokenizers` crates
4. Add Unix socket daemon for amortizing model load cost across invocations

Key decisions: `piguard_002` (Python→ONNX→Rust), `piguard_004` (single-threaded daemon), `piguard_005` (JSON lines protocol), `piguard_012` (academic model over heuristics)

## Key Patterns

**Daemon + auto-detect fallback** (`piguard_003`): CLI tries Unix socket first, falls back to local model load. Zero config — works with or without daemon, performance scales with usage.

**Three-commit architecture** (`piguard_008`): Monolith → extract lib → add features. Each commit is shippable. Avoids premature abstraction while ending up with clean structure.

**Python as setup script only** (`piguard_009`): export_onnx.py is a one-time operation. After that, Python never runs. Clean separation between ML ecosystem (training/export) and serving (Rust).

## What Didn't Work

**Loading model per invocation** (`piguard_011`): Even with Rust's fast cold start (300ms), batch processing 100 lines took 30s. Daemon reduced this to 1.4s — 21x improvement.

Initial implementation was a single main.rs with everything inlined. Worked but couldn't share the Detector between CLI and server. Extracting lib.rs in second commit fixed this cleanly.

## Numbers

| Metric | Value |
|---|---|
| Total Rust LOC | 530 |
| Python setup script | 71 lines |
| Model size (ONNX) | ~735 MB |
| Cold start (Python) | ~2,000 ms |
| Cold start (Rust) | ~300 ms |
| Via daemon | ~14 ms |
| Inference only | ~10 ms |
| Development time | < 1 day (3 commits, same date) |
| Dependencies (Rust) | 6 direct (ort, tokenizers, clap, serde, serde_json, dirs + tracing for server) |

## If I Started Over

1. **Would start with lib.rs + two bins from the start** — the monolith→extract refactor was predictable, could've saved the commit
2. **Would add daemon auto-start** — if daemon isn't running, CLI could fork one in the background instead of loading model locally. Best of both worlds
3. **Would benchmark Python baseline more rigorously** — profiling numbers (800ms import, 400ms load, 500ms init) are estimates, not instrumented

## Open Questions

- How does the model perform on multilingual input? (PIGuard trained on English)
- Would quantization (INT8) reduce the 735MB model size significantly without accuracy loss?
- Is there a lighter model that could eliminate the daemon need entirely (< 50ms cold start)?
- Should the daemon auto-start when CLI detects it's not running?
