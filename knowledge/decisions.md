# Decision Log

## D001: Rust + ONNX Runtime over Python transformers
- **Date:** 2026-03-24
- **Context:** Need fast local prompt injection detection CLI. Python startup is 2s.
- **Options considered:** (A) Python with daemon, (B) Rust with ONNX, (C) C++ with ONNX
- **Chosen:** B — Rust with ONNX
- **Reasoning:** Rust has mature ONNX bindings (ort crate) and HuggingFace tokenizer crate. C++ would work but slower to develop. Python daemon possible but still has import overhead and more complex deployment.
- **Outcome:** 300ms cold start, 14ms via daemon. 530 lines total.
- **Would change?:** No. Sweet spot of development speed and runtime performance.

## D002: PIGuard model over custom classifier or heuristics
- **Date:** 2026-03-24
- **Context:** Need accurate prompt injection detection with low false positives.
- **Options considered:** (A) Regex/keyword matching, (B) Train custom model, (C) Use PIGuard (academic, ACL 2025)
- **Chosen:** C — PIGuard
- **Reasoning:** Academic model with published results. MOF training reduces false positives (key issue with regex). No training data or GPU time needed. DeBERTa-v3-base is small enough for local inference.
- **Outcome:** High accuracy, handles nuance ("ignore" in normal context vs injection).
- **Would change?:** No. Academic models for well-studied problems save weeks.

## D003: Unix socket over HTTP/gRPC for daemon
- **Date:** 2026-03-24
- **Context:** Need IPC between CLI and daemon on same machine.
- **Options considered:** (A) HTTP localhost, (B) gRPC, (C) Unix socket + JSON lines
- **Chosen:** C — Unix socket + JSON lines
- **Reasoning:** Lowest latency (no TCP overhead), simplest implementation, testable with nc/socat. No framework dependency. JSON lines = one serde call per direction.
- **Outcome:** 8-10ms round trip including inference. Scriptable from any language.
- **Would change?:** No. Would only add HTTP if remote access needed.

## D004: Single-threaded daemon over async/threaded
- **Date:** 2026-03-24
- **Context:** Daemon serves local CLI invocations. Typical load: 1 request at a time.
- **Options considered:** (A) tokio async, (B) std::thread per connection, (C) single-threaded blocking
- **Chosen:** C — single-threaded blocking
- **Reasoning:** One user, one machine, sequential CLI calls. Async/threads add complexity for zero benefit. Model inference is CPU-bound anyway — parallelism doesn't help.
- **Outcome:** 180 lines for full server. No async runtime dependency.
- **Would change?:** No, unless building a shared service for multiple users.

## D005: Structured logging (tracing) in daemon only
- **Date:** 2026-03-24
- **Context:** CLI is fire-and-forget, daemon is long-running.
- **Options considered:** (A) No logging, (B) println!, (C) tracing crate
- **Chosen:** C for daemon, no structured logging for CLI
- **Reasoning:** Daemon needs observability (connection tracking, per-request metrics, error context). CLI output IS the user interface — no logging needed. tracing crate adds structured fields without formatting boilerplate.
- **Outcome:** RUST_LOG=debug shows connection IDs, request counts, latency per query. Default level shows classifications only.
- **Would change?:** No. Right amount of observability for the use case.
