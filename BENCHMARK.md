# Benchmark Results

Performance benchmarks for the CLIP image similarity computation pipeline.

## Test Configuration

### Dataset
- **Name**: PIPA Dataset
- **Total Images**: 30,552
- **Task**: Full pairwise distance matrix computation (float16)

### Setup
- **GPU**: NVIDIA RTX 5090 (32 GB VRAM)
- **Batch Size**: 256
- **Model**: `hf-hub:apple/DFN5B-CLIP-ViT-H-14-384`
- **PyTorch**: 2.9.1+cu128
- **CUDA version**: 12.8
- **Open CLIP version**: 3.2.0
- **Precision**: float16 (pairwise distance storage)

### Command
```bash
make run \
  INPUT_DIR=/path/to/PIPA \
  OUTPUT_DIR=./results_pipa \
  MODEL=hf-hub:apple/DFN5B-CLIP-ViT-H-14-384 \
  BATCH_SIZE=256 \
  DEVICE=cuda \
  ANONYMIZE_LABELS=labels.json \
  PAIRWISE_DTYPE=float16
```

## Performance Results

### Timing
| Operation | Time | Percentage |
| --- | --- | --- |
| **Embedding Computation** | 15m 18s (918s) | 96% |
| **Similarity Computation + File I/O** | 39s | 4% |
| **Total Runtime** | 15m 57s (957s) | 100% |

### Throughput
- **Images per second**: 31.92 images/s

### Resource Usage
| Resource | Peak Usage |
| --- | --- |
| GPU Memory (VRAM) | 26 GB |
| System RAM | 20 GB |

### Output
- **File**: `pairwise_distances.npz` (compressed)
- **Size**: 685.4 MB
- **Format**: float16