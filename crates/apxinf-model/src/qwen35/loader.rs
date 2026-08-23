use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use apxinf_loader::safetensors::{read_sharded_tensor_manifest, read_tensor_manifest};
use apxinf_loader::{LoaderManifest, ManifestDType, LOADER_MANIFEST_SCHEMA, QWEN35_MODEL_REVISION};
use thiserror::Error;

use super::attention::PackedLinearReference;
use super::config::{Qwen35ConfigError, Qwen35ModelConfig};
use super::weights::{PackedLinearLayout, WeightLayoutError};

#[derive(Debug, Error)]
pub enum Qwen35LoaderError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("config error: {0}")]
    Config(#[from] Qwen35ConfigError),
    #[error("manifest error: {0}")]
    Manifest(#[from] apxinf_loader::manifest::ManifestError),
    #[error("checkpoint revision must be {expected}, got {actual}")]
    Revision {
        expected: &'static str,
        actual: String,
    },
    #[error("checkpoint inventory is empty")]
    EmptyInventory,
    #[error("unsupported tensor dtype for `{name}`: {dtype:?}")]
    UnsupportedDType { name: String, dtype: ManifestDType },
    #[error("unsupported quantization layout in config: {0}")]
    QuantizationLayout(String),
    #[error("safetensors inventory: {0}")]
    Inventory(String),
    #[error("projection `{base}` is missing tensor `{suffix}`")]
    MissingProjectionTensor { base: String, suffix: &'static str },
    #[error("projection `{base}` tensor `{suffix}` has dtype {dtype:?}, expected {expected}")]
    ProjectionDType {
        base: String,
        suffix: &'static str,
        dtype: ManifestDType,
        expected: &'static str,
    },
    #[error("projection `{base}` has invalid shape metadata: {details}")]
    ProjectionShape { base: String, details: String },
    #[error("projection `{base}` layout validation failed: {source}")]
    ProjectionLayout {
        base: String,
        #[source]
        source: WeightLayoutError,
    },
}

#[derive(Debug, Clone)]
pub struct Qwen35CheckpointInventory {
    pub revision: String,
    pub config: Qwen35ModelConfig,
    pub manifest: LoaderManifest,
    pub inventory_sha256: String,
    pub source_files: BTreeMap<String, String>,
    checkpoint_dir: Option<PathBuf>,
    tensor_shards: BTreeMap<String, String>,
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
        let (tensors, tensor_shards) = if index_path.is_file() {
            let index_raw = std::fs::read_to_string(&index_path)
                .map_err(|e| Qwen35LoaderError::Io(format!("{}: {e}", index_path.display())))?;
            let index: serde_json::Value = serde_json::from_str(&index_raw)
                .map_err(|e| Qwen35LoaderError::Io(format!("{}: {e}", index_path.display())))?;
            let tensor_shards = index
                .get("weight_map")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| Qwen35LoaderError::Inventory("missing weight_map".into()))?
                .iter()
                .map(|(name, shard)| {
                    Ok((
                        name.clone(),
                        shard
                            .as_str()
                            .ok_or_else(|| {
                                Qwen35LoaderError::Inventory("invalid shard name".into())
                            })?
                            .to_owned(),
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, Qwen35LoaderError>>()?;
            (
                read_sharded_tensor_manifest(&index_path).map_err(Qwen35LoaderError::Inventory)?,
                tensor_shards,
            )
        } else {
            let model_path = dir.join("model.safetensors");
            if !model_path.is_file() {
                return Err(Qwen35LoaderError::Io(format!(
                    "missing model.safetensors.index.json or model.safetensors in {}",
                    dir.display()
                )));
            }
            let tensors =
                read_tensor_manifest(&model_path).map_err(Qwen35LoaderError::Inventory)?;
            let tensor_shards = tensors
                .iter()
                .map(|tensor| (tensor.name.clone(), "model.safetensors".to_owned()))
                .collect();
            (tensors, tensor_shards)
        };
        let mut source_files = BTreeMap::new();
        source_files.insert("config.json".into(), sha256_file(&config_path)?);
        if index_path.is_file() {
            source_files.insert(
                "model.safetensors.index.json".into(),
                sha256_file(&index_path)?,
            );
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
        Ok(Self {
            revision,
            config,
            manifest,
            inventory_sha256,
            source_files,
            checkpoint_dir: Some(dir.to_owned()),
            tensor_shards,
        })
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
            checkpoint_dir: None,
            tensor_shards: BTreeMap::new(),
        })
    }

    /// Read one tensor's raw on-disk bytes. No other tensor payload is loaded.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<Vec<u8>, Qwen35LoaderError> {
        let dir = self.checkpoint_dir.as_ref().ok_or_else(|| {
            Qwen35LoaderError::Io("inventory has no checkpoint payload directory".into())
        })?;
        let shard = self
            .tensor_shards
            .get(name)
            .ok_or_else(|| Qwen35LoaderError::Inventory(format!("unknown tensor `{name}`")))?;
        let path = dir.join(shard);
        let mut file = std::fs::File::open(&path)
            .map_err(|e| Qwen35LoaderError::Io(format!("{}: {e}", path.display())))?;
        let file_len = usize::try_from(
            file.metadata()
                .map_err(|e| Qwen35LoaderError::Io(format!("{}: {e}", path.display())))?
                .len(),
        )
        .map_err(|_| Qwen35LoaderError::Inventory("shard length overflow".into()))?;
        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)
            .map_err(|e| Qwen35LoaderError::Inventory(format!("SafeTensors header length: {e}")))?;
        let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| Qwen35LoaderError::Inventory("header length overflow".into()))?;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Qwen35LoaderError::Inventory("header length overflow".into()))?;
        if header_end > file_len {
            return Err(Qwen35LoaderError::Inventory(
                "truncated SafeTensors header".into(),
            ));
        }
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(|e| Qwen35LoaderError::Inventory(format!("SafeTensors header: {e}")))?;
        let header: serde_json::Value = serde_json::from_slice(&header_bytes).map_err(|e| {
            Qwen35LoaderError::Inventory(format!("invalid SafeTensors header: {e}"))
        })?;
        let info = header
            .get(name)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                Qwen35LoaderError::Inventory(format!("tensor `{name}` absent from shard"))
            })?;
        let offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|values| values.len() == 2)
            .ok_or_else(|| {
                Qwen35LoaderError::Inventory(format!("tensor `{name}` has invalid offsets"))
            })?;
        let start = offsets[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                Qwen35LoaderError::Inventory(format!("tensor `{name}` has invalid start"))
            })?;
        let end = offsets[1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                Qwen35LoaderError::Inventory(format!("tensor `{name}` has invalid end"))
            })?;
        let start = header_end
            .checked_add(start)
            .ok_or_else(|| Qwen35LoaderError::Inventory("tensor offset overflow".into()))?;
        let end = header_end
            .checked_add(end)
            .ok_or_else(|| Qwen35LoaderError::Inventory("tensor offset overflow".into()))?;
        if start > end || end > file_len {
            return Err(Qwen35LoaderError::Inventory(format!(
                "tensor `{name}` offsets exceed shard"
            )));
        }
        let mut payload = vec![0u8; end - start];
        file.seek(SeekFrom::Start(start as u64))
            .and_then(|_| file.read_exact(&mut payload))
            .map_err(|e| Qwen35LoaderError::Io(format!("{}: {e}", path.display())))?;
        Ok(payload)
    }

    pub fn tensor_manifest(
        &self,
        name: &str,
    ) -> Result<&apxinf_loader::TensorManifest, Qwen35LoaderError> {
        self.manifest
            .tensor(name)
            .ok_or_else(|| Qwen35LoaderError::Inventory(format!("unknown tensor `{name}`")))
    }

    pub fn read_tensor_u32(&self, name: &str) -> Result<Vec<u32>, Qwen35LoaderError> {
        let manifest = self.tensor_manifest(name)?;
        if manifest.dtype != ManifestDType::I32 {
            return Err(Qwen35LoaderError::UnsupportedDType {
                name: name.to_owned(),
                dtype: manifest.dtype.clone(),
            });
        }
        let bytes = self.read_tensor_bytes(name)?;
        if bytes.len() % 4 != 0 {
            return Err(Qwen35LoaderError::Inventory(format!(
                "I32 tensor `{name}` byte length is not divisible by 4"
            )));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    pub fn read_tensor_bf16_f32(&self, name: &str) -> Result<Vec<f32>, Qwen35LoaderError> {
        let manifest = self.tensor_manifest(name)?;
        if manifest.dtype != ManifestDType::BF16 {
            return Err(Qwen35LoaderError::UnsupportedDType {
                name: name.to_owned(),
                dtype: manifest.dtype.clone(),
            });
        }
        let bytes = self.read_tensor_bytes(name)?;
        if bytes.len() % 2 != 0 {
            return Err(Qwen35LoaderError::Inventory(format!(
                "BF16 tensor `{name}` byte length is not divisible by 2"
            )));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| {
                half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()
            })
            .collect())
    }

    /// Read one asymmetric W4 projection without materializing any other tensor.
    pub fn read_packed_linear(
        &self,
        base: &str,
    ) -> Result<PackedLinearReference, Qwen35LoaderError> {
        let shape_name = format!("{base}.weight_shape");
        let shape_manifest = self.tensor_manifest_or_projection(base, "weight_shape")?;
        if shape_manifest.shape != [2] {
            return Err(Qwen35LoaderError::ProjectionShape {
                base: base.to_owned(),
                details: format!(
                    "weight_shape tensor must have manifest shape [2], got {:?}",
                    shape_manifest.shape
                ),
            });
        }
        if shape_manifest.dtype != ManifestDType::Other("I64".to_owned()) {
            return Err(Qwen35LoaderError::ProjectionDType {
                base: base.to_owned(),
                suffix: "weight_shape",
                dtype: shape_manifest.dtype.clone(),
                expected: "I64",
            });
        }
        let dimensions = self.read_tensor_i64(&shape_name)?;
        if dimensions.len() != 2 || dimensions.iter().any(|value| *value <= 0) {
            return Err(Qwen35LoaderError::ProjectionShape {
                base: base.to_owned(),
                details: format!(
                    "weight_shape payload must contain two positive values, got {dimensions:?}"
                ),
            });
        }
        let out_features =
            usize::try_from(dimensions[0]).map_err(|_| Qwen35LoaderError::ProjectionShape {
                base: base.to_owned(),
                details: format!("output dimension overflows usize: {}", dimensions[0]),
            })?;
        let in_features =
            usize::try_from(dimensions[1]).map_err(|_| Qwen35LoaderError::ProjectionShape {
                base: base.to_owned(),
                details: format!("input dimension overflows usize: {}", dimensions[1]),
            })?;
        let layout = PackedLinearLayout::new(out_features, in_features, 32);
        let packed_name = format!("{base}.weight_packed");
        let scale_name = format!("{base}.weight_scale");
        let zero_point_name = format!("{base}.weight_zero_point");
        let packed_manifest = self.tensor_manifest_or_projection(base, "weight_packed")?;
        let scale_manifest = self.tensor_manifest_or_projection(base, "weight_scale")?;
        let zero_point_manifest = self.tensor_manifest_or_projection(base, "weight_zero_point")?;
        if packed_manifest.dtype != ManifestDType::I32 {
            return Err(Qwen35LoaderError::ProjectionDType {
                base: base.to_owned(),
                suffix: "weight_packed",
                dtype: packed_manifest.dtype.clone(),
                expected: "I32",
            });
        }
        if scale_manifest.dtype != ManifestDType::BF16 {
            return Err(Qwen35LoaderError::ProjectionDType {
                base: base.to_owned(),
                suffix: "weight_scale",
                dtype: scale_manifest.dtype.clone(),
                expected: "BF16",
            });
        }
        if zero_point_manifest.dtype != ManifestDType::I32 {
            return Err(Qwen35LoaderError::ProjectionDType {
                base: base.to_owned(),
                suffix: "weight_zero_point",
                dtype: zero_point_manifest.dtype.clone(),
                expected: "I32",
            });
        }
        layout
            .validate_shapes(
                &packed_manifest.shape,
                &scale_manifest.shape,
                &zero_point_manifest.shape,
            )
            .map_err(|source| Qwen35LoaderError::ProjectionLayout {
                base: base.to_owned(),
                source,
            })?;
        let weight_packed = self.read_tensor_u32(&packed_name)?;
        let scales = self.read_tensor_bf16_f32(&scale_name)?;
        let zero_points = self.read_tensor_u32(&zero_point_name)?;
        if weight_packed.len() != out_features * layout.packed_k_columns()
            || scales.len() != out_features * layout.groups()
            || zero_points.len() != layout.packed_n_rows() * layout.groups()
        {
            return Err(Qwen35LoaderError::ProjectionShape {
                base: base.to_owned(),
                details: "payload lengths do not match manifest shapes".into(),
            });
        }
        if let Some(index) = scales.iter().position(|value| !value.is_finite()) {
            return Err(Qwen35LoaderError::ProjectionShape {
                base: base.to_owned(),
                details: format!("scale payload at index {index} is not finite"),
            });
        }
        Ok(PackedLinearReference {
            layout,
            weight_packed,
            scales,
            zero_points,
        })
    }

    fn tensor_manifest_or_projection(
        &self,
        base: &str,
        suffix: &'static str,
    ) -> Result<&apxinf_loader::TensorManifest, Qwen35LoaderError> {
        let name = format!("{base}.{suffix}");
        self.manifest
            .tensor(&name)
            .ok_or_else(|| Qwen35LoaderError::MissingProjectionTensor {
                base: base.to_owned(),
                suffix,
            })
    }

    fn read_tensor_i64(&self, name: &str) -> Result<Vec<i64>, Qwen35LoaderError> {
        let manifest = self.tensor_manifest(name)?;
        if manifest.dtype != ManifestDType::Other("I64".to_owned()) {
            return Err(Qwen35LoaderError::UnsupportedDType {
                name: name.to_owned(),
                dtype: manifest.dtype.clone(),
            });
        }
        let bytes = self.read_tensor_bytes(name)?;
        if bytes.len() % 8 != 0 {
            return Err(Qwen35LoaderError::Inventory(format!(
                "I64 tensor `{name}` byte length is not divisible by 8"
            )));
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn validate_tensor_inventory(manifest: &LoaderManifest) -> Result<(), Qwen35LoaderError> {
        if manifest.tensors.is_empty() {
            return Err(Qwen35LoaderError::EmptyInventory);
        }
        for tensor in &manifest.tensors {
            let allowed_shape_metadata = matches!(
                (&tensor.dtype, tensor.name.as_str()),
                (ManifestDType::Other(dtype), name)
                    if dtype == "I64" && name.ends_with(".weight_shape")
            );
            if matches!(tensor.dtype, ManifestDType::Other(_)) && !allowed_shape_metadata {
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
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (bytes.len() as u64) * 8;
    let mut msg = bytes.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            (hh, g, f, e, d, c, b, a) = (
                g,
                f,
                e,
                d.wrapping_add(temp1),
                c,
                b,
                a,
                temp1.wrapping_add(temp2),
            );
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::MODEL_VOCAB_SIZE;
    use crate::qwen35::PackedLinearLayout;
    use apxinf_loader::{ManifestDType, TensorManifest};

    fn config() -> String {
        std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/qwen35-metadata/config.json"),
        )
        .unwrap()
    }

    #[test]
    fn manifest_loader_is_revision_and_dtype_fail_closed() {
        let mut manifest = LoaderManifest {
            schema: LOADER_MANIFEST_SCHEMA.into(),
            revision: QWEN35_MODEL_REVISION.into(),
            vocab_size: MODEL_VOCAB_SIZE,
            tensors: vec![TensorManifest {
                name: "embed.weight".into(),
                shape: vec![MODEL_VOCAB_SIZE, 8],
                dtype: ManifestDType::BF16,
                quantization_role: None,
                pack_axis: None,
                group_size: None,
            }],
        };
        let loaded = Qwen35CheckpointInventory::from_manifest(&config(), manifest.clone()).unwrap();
        assert_eq!(loaded.revision, QWEN35_MODEL_REVISION);
        assert_eq!(loaded.inventory_sha256.len(), 64);
        manifest.tensors[0].dtype = ManifestDType::Other("U4".into());
        assert!(matches!(
            Qwen35CheckpointInventory::from_manifest(&config(), manifest),
            Err(Qwen35LoaderError::UnsupportedDType { .. })
        ));
    }

    #[test]
    fn inventory_digest_uses_standard_sha256() {
        assert_eq!(
            hex_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn checkpoint_inventory_reads_one_tensor_payload_without_loading_the_model() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), config()).unwrap();
        let shard_name = "model-00001-of-00001.safetensors";
        let shard = tiny_safetensors("embed_tokens.weight", "F32", &[2], &[1, 2, 3, 4]);
        std::fs::write(directory.path().join(shard_name), shard).unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{"embed_tokens.weight":"{shard_name}"}}}}"#),
        )
        .unwrap();
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(directory.path(), QWEN35_MODEL_REVISION)
                .unwrap();
        assert_eq!(
            inventory.read_tensor_bytes("embed_tokens.weight").unwrap(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn checkpoint_inventory_rejects_unsafe_index_shard_before_payload_access() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), config()).unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"embed_tokens.weight":"../outside.safetensors"}}"#,
        )
        .unwrap();
        let error =
            Qwen35CheckpointInventory::from_checkpoint_dir(directory.path(), QWEN35_MODEL_REVISION)
                .unwrap_err();
        assert!(error.to_string().contains("unsafe shard path"));
    }

    #[test]
    fn typed_payload_views_reject_dtype_mismatch_and_decode_little_endian_values() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), config()).unwrap();
        let shard_name = "model-00001-of-00001.safetensors";
        let shard = tiny_safetensors_multi(&[
            ("packed", "I32", &[2], &[1, 0, 0, 0, 0xef, 0xbe, 0xad, 0xde]),
            ("scale", "BF16", &[1], &[0x00, 0x3f]),
        ]);
        std::fs::write(directory.path().join(shard_name), shard).unwrap();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{"packed":"{shard_name}","scale":"{shard_name}"}}}}"#),
        )
        .unwrap();
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(directory.path(), QWEN35_MODEL_REVISION)
                .unwrap();
        assert_eq!(
            inventory.read_tensor_u32("packed").unwrap(),
            vec![1, 0xdead_beef]
        );
        assert!((inventory.read_tensor_bf16_f32("scale").unwrap()[0] - 0.5).abs() < 1e-6);
        assert!(matches!(
            inventory.read_tensor_u32("scale"),
            Err(Qwen35LoaderError::UnsupportedDType { .. })
        ));
    }

    #[test]
    fn i64_is_allowed_only_for_weight_shape_metadata() {
        let mut manifest = LoaderManifest {
            schema: LOADER_MANIFEST_SCHEMA.into(),
            revision: QWEN35_MODEL_REVISION.into(),
            vocab_size: MODEL_VOCAB_SIZE,
            tensors: vec![TensorManifest {
                name: "layer.weight_shape".into(),
                shape: vec![2],
                dtype: ManifestDType::Other("I64".into()),
                quantization_role: None,
                pack_axis: None,
                group_size: None,
            }],
        };
        assert!(Qwen35CheckpointInventory::from_manifest(&config(), manifest.clone()).is_ok());
        manifest.tensors[0].name = "layer.weight".into();
        assert!(matches!(
            Qwen35CheckpointInventory::from_manifest(&config(), manifest),
            Err(Qwen35LoaderError::UnsupportedDType { .. })
        ));
    }

    #[test]
    #[ignore = "requires the pinned Qwen3.5 checkpoint payload"]
    fn real_checkpoint_reads_a_packed_projection_without_expanding_checkpoint() {
        let dir = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("APXINF_QWEN35_CHECKPOINT must point to the pinned checkpoint");
        let inventory =
            Qwen35CheckpointInventory::from_checkpoint_dir(&dir, QWEN35_MODEL_REVISION).unwrap();
        let projection = inventory
            .read_packed_linear("model.language_model.layers.0.linear_attn.in_proj_qkv")
            .unwrap();
        assert_eq!(
            projection.layout,
            PackedLinearLayout::new(10_240, 5_120, 32)
        );
        assert_eq!(projection.weight_packed.len(), 10_240 * 640);
        assert_eq!(projection.scales.len(), 10_240 * 160);
        assert_eq!(projection.zero_points.len(), 1_280 * 160);
        assert!(projection.scales.iter().all(|value| value.is_finite()));
    }

    fn tiny_safetensors(name: &str, dtype: &str, shape: &[usize], payload: &[u8]) -> Vec<u8> {
        tiny_safetensors_multi(&[(name, dtype, shape, payload)])
    }

    fn tiny_safetensors_multi(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut offset = 0usize;
        let mut entries = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            entries.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":{:?},"data_offsets":[{},{}]}}"#,
                shape,
                offset,
                offset + payload.len()
            ));
            offset += payload.len();
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for (_, _, _, payload) in tensors {
            bytes.extend_from_slice(payload);
        }
        bytes
    }
}
