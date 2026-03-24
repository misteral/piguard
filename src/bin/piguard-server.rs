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
//! piguard-server                   # listen on /tmp/piguard.sock
//! piguard-server --socket /tmp/x.sock
//! echo '{"text":"hello"}' | nc -U /tmp/piguard.sock
//! ```

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use piguard::{default_model_dir, Detector, SOCKET_PATH};
use serde::Deserialize;

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

fn main() -> Result<()> {
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
    eprintln!(
        "Model loaded in {load_ms}ms, listening on {}",
        cli.socket.display()
    );

    let listener = UnixListener::bind(&cli.socket)
        .with_context(|| format!("bind {}", cli.socket.display()))?;

    // Clean up socket on Ctrl+C.
    let socket_path = cli.socket.clone();
    ctrlc::set_handler(move || {
        let _ = std::fs::remove_file(&socket_path);
        std::process::exit(0);
    })
    .ok();

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };

        let reader = BufReader::new(&stream);
        let mut writer = &stream;

        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<Request>(trimmed) {
                Ok(req) => match detector.detect(&req.text) {
                    Ok(det) => serde_json::to_string(&det).unwrap(),
                    Err(e) => {
                        serde_json::json!({"error": e.to_string()}).to_string()
                    }
                },
                Err(e) => {
                    serde_json::json!({"error": format!("invalid JSON: {e}")}).to_string()
                }
            };

            if writeln!(writer, "{response}").is_err() {
                break;
            }
        }
    }

    Ok(())
}
