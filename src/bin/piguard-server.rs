//! Unix socket server for prompt injection detection.
//!
//! Keeps the model loaded in memory, accepts newline-delimited JSON
//! requests, and returns JSON responses. Designed for low-latency
//! integration with other processes.
//!
//! # Protocol
//!
//! Request (one JSON object per line):
//! ```json
//! {"text": "Ignore previous instructions"}
//! ```
//!
//! Response (one JSON object per line):
//! ```json
//! {"text":"Ignore previous instructions","label":"INJECTION","score":0.999,"latency_ms":10.2}
//! ```
//!
//! # Usage
//!
//! ```bash
//! piguard-server                          # default, info level
//! RUST_LOG=debug piguard-server           # verbose
//! RUST_LOG=warn piguard-server            # quiet
//! echo '{"text":"hello"}' | nc -U /tmp/piguard.sock
//! ```

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use piguard::{default_model_dir, Detector, SOCKET_PATH};
use serde::Deserialize;
use tracing::{debug, error, info, warn};

/// Monotonic connection counter.
static CONN_ID: AtomicU64 = AtomicU64::new(1);

/// `PIGuard` Unix socket server — model stays in memory.
#[derive(Parser, Debug)]
#[command(name = "piguard-server", about, version)]
struct Cli {
    /// Path to the Unix socket.
    #[arg(long, default_value = SOCKET_PATH)]
    socket: PathBuf,

    /// Path to model directory.
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

/// Incoming JSON request.
#[derive(Debug, Deserialize)]
struct Request {
    text: String,
}

#[expect(clippy::needless_pass_by_value, reason = "stream is consumed on drop")]
fn handle_connection(stream: UnixStream, detector: &mut Detector, conn_id: u64) {
    debug!(conn_id, "connection accepted");

    let reader = BufReader::new(&stream);
    let mut writer = &stream;
    let mut req_count: u64 = 0;

    for line in reader.lines() {
        let Ok(line) = line else {
            debug!(conn_id, "read error, closing");
            break;
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        req_count += 1;

        let response = match serde_json::from_str::<Request>(trimmed) {
            Ok(req) => {
                let text_preview: String = req.text.chars().take(80).collect();
                match detector.detect(&req.text) {
                    Ok(det) => {
                        info!(
                            conn_id,
                            req_n = req_count,
                            label = %det.label,
                            score = det.score,
                            latency_ms = format_args!("{:.1}", det.latency_ms),
                            text = %text_preview,
                            "classified"
                        );
                        serde_json::to_string(&det).unwrap()
                    }
                    Err(e) => {
                        error!(
                            conn_id,
                            err = %e,
                            text = %text_preview,
                            "inference failed"
                        );
                        serde_json::json!({"error": e.to_string()}).to_string()
                    }
                }
            }
            Err(e) => {
                warn!(conn_id, err = %e, "invalid JSON request");
                serde_json::json!({"error": format!("invalid JSON: {e}")}).to_string()
            }
        };

        if writeln!(writer, "{response}").is_err() {
            debug!(conn_id, "write error, closing");
            break;
        }
    }

    debug!(conn_id, requests = req_count, "connection closed");
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "piguard_server=info,warn".parse().unwrap()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let model_dir = cli.model_dir.unwrap_or_else(default_model_dir);

    // Clean up stale socket.
    if cli.socket.exists() {
        std::fs::remove_file(&cli.socket)
            .with_context(|| format!("remove stale socket {}", cli.socket.display()))?;
    }

    // Load model once.
    let load_start = Instant::now();
    let mut detector = Detector::load(&model_dir)?;
    let load_ms = load_start.elapsed().as_millis();

    info!(
        socket = %cli.socket.display(),
        model_dir = %model_dir.display(),
        load_ms = %load_ms,
        "server ready"
    );

    let listener = UnixListener::bind(&cli.socket)
        .with_context(|| format!("bind {}", cli.socket.display()))?;

    // Clean up socket on Ctrl+C.
    let socket_path = cli.socket.clone();
    ctrlc::set_handler(move || {
        info!("shutting down");
        let _ = std::fs::remove_file(&socket_path);
        std::process::exit(0);
    })
    .ok();

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let conn_id = CONN_ID.fetch_add(1, Ordering::Relaxed);
                handle_connection(s, &mut detector, conn_id);
            }
            Err(e) => {
                error!(err = %e, "failed to accept connection");
            }
        }
    }

    Ok(())
}
