//! `PIGuard` — fast local prompt injection detection.
//!
//! Wraps the `PIGuard` ONNX model (DeBERTa-v3-base, ACL 2025) for
//! binary classification of text into `BENIGN` or `INJECTION`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// `HuggingFace` model identifier used for tokenizer download.
pub const MODEL_ID: &str = "leolee99/PIGuard";

/// Default Unix socket path for the daemon.
pub const SOCKET_PATH: &str = "/tmp/piguard.sock";

/// Classification result for a single text.
#[derive(Debug, Serialize, Deserialize)]
pub struct Detection {
    pub text: String,
    pub label: String,
    pub score: f32,
    pub latency_ms: f64,
}

/// ONNX-backed prompt injection detector.
///
/// Loads the model once and reuses it for all subsequent queries.
#[derive(Debug)]
pub struct Detector {
    session: Session,
    tokenizer: Tokenizer,
}

impl Detector {
    /// Load detector from a directory containing `model.onnx` and `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer files are missing or corrupt.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let onnx_path = model_dir.join("model.onnx");
        if !onnx_path.exists() {
            anyhow::bail!(
                "ONNX model not found at {}\nRun: python scripts/export_onnx.py",
                onnx_path.display()
            );
        }

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "tokenizer.json not found at {}",
                tokenizer_path.display()
            );
        }

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
            .commit_from_file(&onnx_path)
            .map_err(|e| anyhow::anyhow!("load ONNX: {e}"))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        Ok(Self { session, tokenizer })
    }

    /// Classify a single text as `BENIGN` or `INJECTION`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    ///
    /// # Panics
    ///
    /// Panics if the model returns no logits (should never happen with a valid model).
    pub fn detect(&mut self, text: &str) -> Result<Detection> {
        let start = Instant::now();

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization: {e}"))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let len = ids.len();

        let ids_vec: Vec<i64> = ids.iter().map(|&x| i64::from(x)).collect();
        let mask_vec: Vec<i64> = mask.iter().map(|&x| i64::from(x)).collect();

        let input_ids = Tensor::from_array(([1usize, len], ids_vec))
            .map_err(|e| anyhow::anyhow!("tensor: {e}"))?;
        let attention_mask = Tensor::from_array(([1usize, len], mask_vec))
            .map_err(|e| anyhow::anyhow!("tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![input_ids, attention_mask])
            .map_err(|e| anyhow::anyhow!("inference: {e}"))?;

        let logits_ref = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract: {e}"))?;

        let logits = logits_ref.1;

        // Softmax over 2 logits.
        let max_val: f32 = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

        let (pred_idx, &score) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // id2label: 0 = benign, 1 = injection
        let label = if pred_idx == 1 { "INJECTION" } else { "BENIGN" }.to_owned();
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(Detection {
            text: text.to_string(),
            label,
            score,
            latency_ms,
        })
    }
}

/// Resolve the default model directory (`~/.cache/piguard/onnx`).
#[must_use] 
pub fn default_model_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("piguard")
        .join("onnx")
}
