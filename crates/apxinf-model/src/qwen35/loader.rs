use std::collections::BTreeMap;
use std::path::Path;

use apxinf_loader::{
    LoaderManifest, ManifestDType, LOADER_MANIFEST_SCHEMA, QWEN35_MODEL_REVISION,
};
use apxinf_loader::safetensors::{read_sharded_tensor_manifest, read_tensor_manifest};
use thiserror::Error;

use super::config::{Qwen35ConfigError, Qwen35ModelConfig};

#[derive(Debug, Error)]
pub enum Qwen35LoaderError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("config error: {0}")]
    Config(#[from] Qwen35ConfigError),
    #[error("manifest error: {0}")]
    Manifest(#[from] apxinf_loader::manifest::ManifestError),
    #[error("checkpoint revision must be {expected}, got {actual}")]
    Revision { expected: &'static str, actual: String },
    #[error("checkpoint inventory is empty")]
    EmptyInventory,
    #[error("unsupported tensor dtype for `{name}`: {dtype:?}")]
    UnsupportedDType { name: String, dtype: ManifestDType },
    #[error("unsupported quantization layout in config: {0}")]
    QuantizationLayout(String),
    #[error("safetensors inventory: {0}")]
    Inventory(String),
}

#[derive(Debug, Clone)]
pub struct Qwen35CheckpointInventory {
    pub revision: String,
    pub config: Qwen35ModelConfig,
    pub manifest: LoaderManifest,
    pub inventory_sha256: String,
    pub source_files: BTreeMap<String, String>,
}

impl Qwen35CheckpointInventory {
    pub fn from_checkpoint_dir(
        dir: impl AsRef<Path>,
        revision: impl Into<String>,
    ) -> Result<Self, Qwen35LoaderError> {
        let dir = dir.as_ref();
        let revision = revision.into();
        let config_path = dir.join("config.json");
        let config_raw = std::fs::read_to_string(&config_path)
            .map_err(|e| Qwen35LoaderError::Io(format!("{}: {e}", config_path.display())))?;
        let config = Qwen35ModelConfig::from_json_str(&config_raw)?;
        let index_path = dir.join("model.safetensors.index.json");
        let tensors = if index_path.is_file() {
            read_sharded_tensor_manifest(&index_path).map_err(Qwen35LoaderError::Inventory)?
        } else {
            let model_path = dir.join("model.safetensors");
            if !model_path.is_file() {
                return Err(Qwen35LoaderError::Io(format!(
                    "missing model.safetensors.index.json or model.safetensors in {}",
                    dir.display()
                )));
            }
            read_tensor_manifest(&model_path).map_err(Qwen35LoaderError::Inventory)?
        };
        let mut source_files = BTreeMap::new();
        source_files.insert("config.json".into(), sha256_file(&config_path)?);
        if index_path.is_file() {
            source_files.insert("model.safetensors.index.json".into(), sha256_file(&index_path)?);
        }
        let manifest = LoaderManifest {
            schema: LOADER_MANIFEST_SCHEMA.to_owned(),
            revision: revision.clone(),
            vocab_size: config.vocab_size,
            tensors,
        };
        Self::validate_tensor_inventory(&manifest)?;
        manifest.validate()?;
        let inventory_sha256 = hash_manifest(&manifest)?;
        Ok(Self { revision, config, manifest, inventory_sha256, source_files })
    }

    pub fn from_manifest(
        config_json: &str,
        manifest: LoaderManifest,
    ) -> Result<Self, Qwen35LoaderError> {
        let config = Qwen35ModelConfig::from_json_str(config_json)?;
        if manifest.revision != QWEN35_MODEL_REVISION {
            return Err(Qwen35LoaderError::Revision {
                expected: QWEN35_MODEL_REVISION,
                actual: manifest.revision.clone(),
            });
        }
        if manifest.vocab_size != config.vocab_size {
            return Err(Qwen35LoaderError::Manifest(
                apxinf_loader::manifest::ManifestError::VocabSize {
                    expected: config.vocab_size,
                    actual: manifest.vocab_size,
                },
            ));
        }
        Self::validate_tensor_inventory(&manifest)?;
        manifest.validate()?;
        let inventory_sha256 = hash_manifest(&manifest)?;
        Ok(Self {
            revision: manifest.revision.clone(),
            config,
            manifest,
            inventory_sha256,
            source_files: BTreeMap::new(),
        })
    }

    fn validate_tensor_inventory(manifest: &LoaderManifest) -> Result<(), Qwen35LoaderError> {
        if manifest.tensors.is_empty() {
            return Err(Qwen35LoaderError::EmptyInventory);
        }
        for tensor in &manifest.tensors {
            if matches!(tensor.dtype, ManifestDType::Other(_)) {
                return Err(Qwen35LoaderError::UnsupportedDType {
                    name: tensor.name.clone(),
                    dtype: tensor.dtype.clone(),
                });
            }
        }
        Ok(())
    }
}

fn hash_manifest(manifest: &LoaderManifest) -> Result<String, Qwen35LoaderError> {
    let encoded = serde_json::to_vec(manifest)
        .map_err(|e| Qwen35LoaderError::Io(format!("manifest serialization: {e}")))?;
    Ok(hex_sha256(&encoded))
}

fn sha256_file(path: &Path) -> Result<String, Qwen35LoaderError> {
    let bytes = std::fs::read(path)
        .map_err(|e| Qwen35LoaderError::Io(format!("{}: {e}", path.display())))?;
    Ok(hex_sha256(&bytes))
}

fn hex_sha256(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ];
    let mut h = [0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
    let bit_len = (bytes.len() as u64) * 8;
    let mut msg = bytes.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes(chunk[i*4..i*4+4].try_into().unwrap()); }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh) = (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            (hh,g,f,e,d,c,b,a) = (g,f,e,d.wrapping_add(temp1),c,b,a,temp1.wrapping_add(temp2));
        }
        for (x,y) in h.iter_mut().zip([a,b,c,d,e,f,g,hh]) { *x = x.wrapping_add(y); }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_loader::{ManifestDType, TensorManifest};
    use crate::qwen35::config::MODEL_VOCAB_SIZE;

    fn config() -> String {
        std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen35-metadata/config.json"),
        ).unwrap()
    }

    #[test]
    fn manifest_loader_is_revision_and_dtype_fail_closed() {
        let mut manifest = LoaderManifest {
            schema: LOADER_MANIFEST_SCHEMA.into(),
            revision: QWEN35_MODEL_REVISION.into(),
            vocab_size: MODEL_VOCAB_SIZE,
            tensors: vec![TensorManifest {
                name: "embed.weight".into(), shape: vec![MODEL_VOCAB_SIZE, 8],
                dtype: ManifestDType::BF16, quantization_role: None, pack_axis: None, group_size: None,
            }],
        };
        let loaded = Qwen35CheckpointInventory::from_manifest(&config(), manifest.clone()).unwrap();
        assert_eq!(loaded.revision, QWEN35_MODEL_REVISION);
        assert_eq!(loaded.inventory_sha256.len(), 64);
        manifest.tensors[0].dtype = ManifestDType::Other("U4".into());
        assert!(matches!(Qwen35CheckpointInventory::from_manifest(&config(), manifest), Err(Qwen35LoaderError::UnsupportedDType { .. })));
    }

    #[test]
    fn inventory_digest_uses_standard_sha256() {
        assert_eq!(hex_sha256(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
