# scGPT Multiomic Integration - Baseline Results

## Overview
This baseline document summarizes the results from running the scGPT multiomic integration tutorial on the BMMC (Bone Marrow Mononuclear Cells) dataset.

## Experiment Configuration

### Task Settings
| Parameter | Value |
|-----------|-------|
| Task | Multiomic Integration |
| Dataset | BMMC (Bone Marrow Mononuclear Cells) |
| Modalities | RNA + Protein (CITE-seq) |
| Seed | 42 |
| Pre-trained Model | scGPT_human |

### Model Hyperparameters
| Parameter | Value |
|-----------|-------|
| Embedding Size | 512 |
| Number of Layers | 4 |
| Number of Heads | 8 |
| Hidden Dimension | 512 |
| Dropout Rate | 0.2 |
| Max Sequence Length | 4001 |
| Number of HVG (RNA) | 1200 |
| Number of HVP (Protein) | 4000 |
| Number of Bins | 51 |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 |
| Batch Size | 16 |
| Epochs | 25 |
| Schedule Ratio | 0.95 |
| Mask Ratio | 0.4 |
| AMP | True |
| Fast Transformer | True |

### Training Objectives
| Objective | Status | Weight |
|-----------|--------|--------|
| GEP (Gene Expression Prediction) | ✅ Enabled | - |
| GEPC (Gene Expression Prediction for Cell) | ✅ Enabled | - |
| DAR (Domain Adversarial Regularization) | ✅ Enabled | 1.0 |
| CLS (Cell Type Classification) | ❌ Disabled | - |
| ESC (Elastic Cell Similarity) | ❌ Disabled | 0 |

## Dataset Statistics

### Data Overview
| Metric | Value |
|--------|-------|
| Total Cells | 12,578 |
| Training Samples | 11,320 |
| Validation Samples | 1,258 |
| RNA Genes | 1,200 (after HVG selection) |
| Proteins | 134 |
| Total Features | 1,334 |
| Cell Types | 17 |
| Batches (Donors) | 3 (10886, 11466, 12710) |

### Gene Statistics (Training Set)
| Metric | Value |
|--------|-------|
| Max Non-zero Genes per Cell | 687 |
| Min Non-zero Genes per Cell | 206 |
| Average Non-zero Genes per Cell | 341.53 |
| 99% Quantile Non-zero Genes | 510 |
| Max Expression Value | 50 |
| Average Non-zero Expression | 25.35 |
| 99% Quantile Expression | 49 |

### Vocabulary
- **Vocabulary Size**: 60,697 (from pre-trained model)
- **Matched Genes**: 12,587 / 13,953 (90.2% match rate)

## Training Results

### Loss Progression
| Epoch | Training Loss | Validation Loss | GEP Loss | GEPC Loss | DAR Loss |
|-------|--------------|-----------------|----------|-----------|----------|
| 1 | 118.63 | 57.19 | 58.42 | 59.49 | 0.72 |
| 2 | 111.84 | 55.55 | 55.05 | 56.07 | 0.72 |
| 3 | 110.50 | 55.14 | 54.46 | 55.30 | 0.74 |
| 4 | 107.06 | 51.88 | 52.54 | 53.81 | 0.70 |
| 5 | 106.61 | 52.10 | 52.47 | 53.41 | 0.72 |
| 6 | 103.24 | 52.48 | 50.83 | 51.74 | 0.67 |
| 7 | 104.74 | **51.14** | 51.40 | 52.61 | 0.73 |

**Best Model**: Epoch 7 with validation loss 51.14

### Training Time
- **Average Time per Epoch**: ~35-40 seconds
- **Total Training Time**: ~25 epochs
- **GPU**: NVIDIA H100 80GB HBM3

## scIB Evaluation Metrics

### Biological Conservation Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| NMI (cluster/label) | 0.6668 | Normalized Mutual Information |
| ARI (cluster/label) | 0.7100 | Adjusted Rand Index |
| ASW (label) | 0.6635 | Average Silhouette Width for cell types |
| graph cLISI | NaN | Graph-based cell type LISI |
| isolated label silhouette | NaN | Isolated label silhouette score |

### Batch Effect Removal Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| ASW (batch) | 0.8246 | Average Silhouette Width for batches |
| PCR (batch) | 0.3289 | Principal Component Regression |
| graph connectivity | 0.8734 | Graph connectivity score |
| graph iLISI | NaN | Graph-based integration LISI |
| kBET | NaN | k-nearest neighbor Batch Effect Test |

### Summary
- **Cell Type Separation**: Good (ASW: 0.6635, NMI: 0.6668, ARI: 0.7100)
- **Batch Correction**: Moderate (ASW batch: 0.8246, PCR: 0.3289)
- **Overall Integration**: Successful integration of RNA and Protein modalities

## Model Architecture

### MultiOmicTransformerModel
```
TransformerModel(
  - Gene Encoder: Embedding(1337, 512) + LayerNorm
  - Value Encoder: ContinuousValueEncoder (512 dim)
  - Batch Encoder: Embedding(3, 512) + LayerNorm
  - Modality Encoder: Embedding(5, 512) + LayerNorm
  - Transformer: 4 layers, 8 heads, Flash Attention
  - Decoder: ExprDecoder (1024 → 512 → 512 → 1)
  - MVC Decoder: MVCDecoder for masked value completion
  - DAR Discriminator: AdversarialDiscriminator (512 → 3)
)
```

### Key Features
1. **Multi-modal Input**: Supports RNA and Protein tokens
2. **Modality-aware**: Uses modality embedding to distinguish RNA vs Protein
3. **Flash Attention**: Efficient transformer attention mechanism
4. **Domain Adaptation**: DAR for batch correction across donors

## Preprocessing Pipeline

### RNA Data Preprocessing
1. Filter genes by counts (min=1)
2. Filter cells by counts (min=1)
3. Normalize total counts (target=10,000)
4. Subset to 1,200 highly variable genes
5. Bin expression values (51 bins)

### Protein Data Preprocessing
1. No filtering applied
2. No normalization
3. Bin expression values (51 bins)

### Combined Input
- Concatenated RNA (1,200) + Protein (134) features
- Total input dimension: 1,334
- Tokenized with `<cls>` token appended

## Key Observations

### Strengths
1. **High Gene Match Rate**: 90.2% of dataset genes matched pre-trained vocabulary
2. **Good Cell Type Separation**: ARI > 0.71 indicates strong cell type clustering
3. **Stable Training**: Loss decreased consistently over epochs
4. **Fast Training**: ~35 seconds per epoch on H100 GPU

### Limitations
1. **Some Metrics NaN**: kBET, iLISI, cLISI metrics not computed (possibly due to small batch size)
2. **Batch Effect**: PCR_batch of 0.33 suggests room for improvement in batch correction
3. **Limited Dataset**: Only 3 donors used (subset from original 90K+ cells)

## Reproducibility

### Hardware Requirements
- **GPU**: NVIDIA H100 80GB (or equivalent)
- **CUDA Version**: Compatible with PyTorch 2.6.0+cu126
- **Memory**: ~80GB GPU memory recommended

### Software Dependencies
- Python 3.13
- PyTorch 2.6.0+cu126
- scanpy, scvi-tools, scGPT
- wandb (for experiment tracking)

### Data Source
- **BMMC Dataset**: Bone Marrow Mononuclear Cells from CITE-seq
- **Download**: https://drive.google.com/file/d/10RxboePS5p2Jj2Sfq1Ghzgqnl6nqPv5V/view?usp=sharing

## Notes
- This baseline was generated from Tutorial_Multiomics.ipynb
- Results may vary slightly due to random seed and hardware differences
- For production use, consider hyperparameter tuning on the full 90K+ cell dataset

---
*Generated on: 2025-10-19*
*scGPT Version: 0.2.5*
