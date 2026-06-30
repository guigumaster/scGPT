#!/usr/bin/env python3
"""
Initialize the scGPT_human pretrained model checkpoint directory.

Creates:
  - args.json: Model configuration
  - vocab.json: Gene vocabulary (comprehensive human gene set)
  - best_model.pt: Initialized model weights (random, not pretrained)

This allows the training script to load the model architecture even when
the actual pretrained checkpoint is not available.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scgpt.tokenizer.gene_tokenizer import GeneVocab


def create_vocab_json(save_dir: Path):
    """Create a comprehensive human gene vocabulary JSON file."""
    vocab_file = save_dir / "vocab.json"
    
    # Common human genes for scRNA-seq (top 2000+ genes)
    common_genes = [
        "MALAT1", "TMSB4X", "TMSB10", "EEF1A1", "GAPDH", "ACTB", "ACTG1",
        "FTL", "FTH1", "B2M", "RPLP0", "RPS18", "RPS27A", "RPS3", "RPS6",
        "RPL13A", "RPL18", "RPL3", "RPL4", "RPL5", "RPL7", "RPL7A", "RPL8",
        "RPL9", "RPL10", "RPL10A", "RPL11", "RPL12", "RPL13", "RPL14", "RPL15",
        "RPL17", "RPL18A", "RPL19", "RPL21", "RPL22", "RPL23", "RPL23A", "RPL24",
        "RPL26", "RPL27", "RPL27A", "RPL28", "RPL29", "RPL30", "RPL31", "RPL32",
        "RPL34", "RPL35", "RPL35A", "RPL36", "RPL36A", "RPL37", "RPL37A", "RPL38",
        "RPL39", "RPL41", "RPS2", "RPS3A", "RPS4X", "RPS4Y1", "RPS5", "RPS7",
        "RPS8", "RPS9", "RPS10", "RPS11", "RPS12", "RPS13", "RPS14", "RPS15",
        "RPS15A", "RPS16", "RPS17", "RPS19", "RPS20", "RPS21", "RPS23", "RPS24",
        "RPS25", "RPS26", "RPS27", "RPS28", "RPS29", "RPSA",
        "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "CD14", "CD19", "CD20",
        "CD34", "CD44", "CD45", "CD68", "CD74", "CD79A", "CD79B", "CD80", "CD86",
        "CD163", "CD207", "CD274",
        "PTPRC", "MS4A1", "NKG7", "GNLY", "GZMB", "GZMA", "GZMK", "PRF1",
        "KLRB1", "KLRC1", "KLRD1", "KLRK1", "NCR1", "NCAM1",
        "FCGR3A", "FCGR3B", "FCRLA", "FCRL1", "FCRL2", "FCRL3", "FCRL4", "FCRL5",
        "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1",
        "HLA-DPA1", "HLA-DPB1", "HLA-E", "HLA-F", "HLA-G",
        "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "CDKN2C", "CDKN2D",
        "TP53", "MYC", "JUN", "FOS", "FOSB", "JUNB", "JUND",
        "BCL2", "BCL2A1", "BCL2L1", "BCL2L11", "BAX", "BAK1", "MCL1",
        "IL2", "IL4", "IL6", "IL7", "IL10", "IL12A", "IL12B", "IL15", "IL18",
        "IL1B", "IL2RA", "IL2RB", "IL2RG", "IL4R", "IL6R", "IL7R", "IL10RA",
        "IFNG", "IFNAR1", "IFNAR2", "IFNGR1", "IFNGR2",
        "TNF", "TNFAIP3", "TNFRSF1A", "TNFRSF1B", "TNFRSF4", "TNFRSF9",
        "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "TGFBR3",
        "CCL2", "CCL3", "CCL4", "CCL5", "CCL19", "CCL20", "CCL21", "CCL22",
        "CXCL8", "CXCL9", "CXCL10", "CXCL11", "CXCL12", "CXCL13",
        "CCR5", "CCR7", "CCR9", "CXCR3", "CXCR4", "CXCR5", "CXCR6",
        "VIM", "CDH1", "CDH2", "EPCAM", "KRT19", "KRT18", "KRT8", "KRT7",
        "MKI67", "PCNA", "TOP2A", "CENPF", "AURKA", "AURKB", "PLK1",
        "SOX4", "SOX9", "SOX2", "POU5F1", "NANOG", "KLF4", "MYOD1",
        "FLT3", "KIT", "CSF1R", "CSF2RA", "CSF3R", "EPOR", "MPL",
        "GATA1", "GATA2", "GATA3", "GATA4", "GATA6",
        "PAX5", "PAX6", "PAX7", "PAX8", "PAX9",
        "FOXP3", "FOXO1", "FOXO3", "FOXA1", "FOXA2",
        "STAT1", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6",
        "NFKB1", "NFKB2", "RELA", "RELB", "REL",
        "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4",
        "WNT1", "WNT2", "WNT3", "WNT3A", "WNT4", "WNT5A", "WNT5B",
        "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3A", "GSK3B",
        "MAPK1", "MAPK3", "MAPK8", "MAPK9", "MAPK10", "MAPK11", "MAPK12",
        "AKT1", "AKT2", "AKT3", "MTOR", "RPTOR", "RICTOR",
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PTEN",
        "KRAS", "HRAS", "NRAS", "BRAF", "ARAF", "RAF1",
        "EGFR", "ERBB2", "ERBB3", "ERBB4",
        "VEGFA", "VEGFB", "VEGFC", "VEGFD", "KDR", "FLT1", "FLT4",
        "EZH2", "SUZ12", "EED", "RING1", "BMI1", "CBX2", "CBX4", "CBX6", "CBX7", "CBX8",
        "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6", "HDAC7", "HDAC8", "HDAC9", "HDAC10", "HDAC11",
        "DNMT1", "DNMT3A", "DNMT3B", "TET1", "TET2", "TET3",
        "MEF2C", "MEIS1", "HOXA9", "HOXA10", "HOXB4", "HOXB5",
        "RUNX1", "RUNX2", "RUNX3", "CBFB",
        "MYB", "MYBL1", "MYBL2", "ETV6", "ERG", "FLI1",
        "CEBPA", "CEBPB", "CEBPD", "CEBPE", "CEBPG",
        "IRF1", "IRF2", "IRF3", "IRF4", "IRF5", "IRF6", "IRF7", "IRF8", "IRF9",
        "EOMES", "TBX21", "RORC", "RORA", "RORB",
        "NEAT1", "NORAD", "HOTAIR", "HOTTIP",
        "SCGB1A1", "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "SFTPD",
        "ALB", "AFP", "CYP2E1", "CYP3A4", "CYP1A2", "CYP2D6",
        "INS", "GCG", "SST", "PPY", "GHRL",
        "MYH6", "MYH7", "MYL2", "MYL3", "MYL7", "TNNT2", "TNNI3", "ACTC1",
        "NPPA", "NPPB", "ACTN2", "DES",
        "SYP", "NEFL", "NEFM", "NEFH", "MAP2", "TUBB3", "RBFOX3",
        "GFAP", "OLIG1", "OLIG2", "MBP", "PLP1", "MOG",
        "AQP4", "SLC1A3", "SLC1A2", "SLC17A6", "SLC17A7", "GAD1", "GAD2",
        "SNAP25", "SYT1", "SYN1", "DLG4", "GRIA1", "GRIN1", "GRIN2A",
        "PDX1", "NKX2-1", "NKX2-2", "NKX6-1", "ISL1", "LMX1A",
        "PECAM1", "VWF", "CDH5", "PODXL", "TEK", "ANGPT1", "ANGPT2",
        "PDGFRA", "PDGFRB", "COL1A1", "COL1A2", "COL3A1", "FN1", "LAMA1", "LAMA2",
        "ITGAM", "ITGAX", "ITGAL", "ITGB1", "ITGB2", "ITGB3", "ITGB7",
        "S100A8", "S100A9", "S100A4", "S100A6", "S100A10", "S100A11", "S100B",
        "LYZ", "CST3", "CSTA", "CSTB", "CTSS", "CTSB", "CTSD",
        "AIF1", "CD86", "IL1B", "TNF", "IL6", "CCL2", "CCL3", "CCL4",
        "MRC1", "CD163", "MSR1", "MARCO", "FCGR1A", "FCGR1B",
        "SPP1", "MGLL", "FABP4", "FABP5", "LPL", "LIPE",
        "DCN", "LUM", "COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "COL6A3",
        "ACTA2", "MYH11", "CNN1", "TAGLN", "MYOCD",
        "NRXN1", "NRXN2", "NRXN3", "NLGN1", "NLGN2", "NLGN3",
        "DMD", "DYSF", "CAPN3", "SGCA", "SGCB", "SGCD", "SGCG",
        "CFTR", "MUC1", "MUC2", "MUC4", "MUC5AC", "MUC5B", "MUC6", "MUC16",
        "TTN", "NEB", "RBM20", "LMNA", "EMD", "LDB3",
        "MET", "FGFR1", "FGFR2", "FGFR3", "FGFR4",
        "PD1", "CTLA4", "LAG3", "TIM3", "TIGIT", "BTLA", "CD28", "ICOS", "GITR", "OX40",
        "ARG1", "ARG2", "NOS2", "IDO1",
        "HBB", "HBA1", "HBA2", "HBD", "HBE1", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ",
        "ALAS2", "EPB42", "SPTB", "SPTA1", "ANK1", "SLC4A1",
        "PF4", "PPBP", "GP1BA", "GP9", "ITGA2B", "ITGB3",
        "CLEC4C", "LILRA4", "TCF4", "IRF8", "JCHAIN", "BCL11A",
        "MZB1", "SDC1", "XBP1", "DERL3", "SSR4", "PPIB",
        "FCN1", "CSF3R",
        "CLEC10A", "FCER1A", "CLEC4A", "CD1C", "CD1E",
        "IL3RA", "LILRA4", "IRF7",
        "HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA1B", "HSPA8", "HSPB1",
        "DNAJB1", "DNAJA1", "DNAJC7", "HSPE1", "HSPD1",
        "UBB", "UBC", "UBD", "UBE2D1", "UBE2D2", "UBE2D3", "UBE2D4",
        "PSMB5", "PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10",
        "EEF2", "EIF4A1", "EIF4E", "EIF4G1", "EIF2S1", "EIF2S2", "EIF2S3",
        "NPM1", "NCL", "NOLC1", "FBL", "DKC1",
        "HNRNPA1", "HNRNPA2B1", "HNRNPC", "HNRNPK", "HNRNPU",
        "DDX5", "DDX17", "DDX21", "DHX9", "DHX15", "DHX30",
        "SNRPN", "SNRPA", "SNRPB", "SNRPD1", "SNRPD2", "SNRPD3",
        "SF3A1", "SF3A2", "SF3A3", "SF3B1", "SF3B2", "SF3B3", "SF3B4", "SF3B5",
        "TRA2A", "TRA2B", "SRSF1", "SRSF2", "SRSF3", "SRSF4", "SRSF5", "SRSF6", "SRSF7",
        "PTMA", "STMN1", "HMGB1", "HMGB2", "HMGN1", "HMGN2",
        "TUBB", "TUBB2A", "TUBB2B", "TUBB3", "TUBB4A", "TUBB4B", "TUBB6",
        "TUBA1A", "TUBA1B", "TUBA1C", "TUBA4A",
        "HIST1H4C", "HIST1H2BK", "HIST1H2BJ", "HIST1H2AC", "HIST1H1C",
        "H2AFX", "H2AFY", "H2AFZ", "H3F3A", "H3F3B",
        "LDHA", "LDHB", "PGK1", "PKM", "ENO1", "GPI", "PFKL", "PFKP",
        "TKT", "TALDO1", "G6PD", "PGD", "GLO1",
        "SOD1", "SOD2", "CAT", "GPX1", "GPX4", "GSTP1", "GSTM1",
        "PRDX1", "PRDX2", "PRDX3", "PRDX4", "PRDX5", "PRDX6",
        "TXNRD1", "TXN", "TXNL1", "GSS", "GSR",
        "MIF", "TIMP1", "TIMP2", "SERPINE1", "SERPINA1", "SERPINB1",
        "MMP9", "MMP2", "MMP14", "ADAM10", "ADAM17",
        "ANXA1", "ANXA2", "ANXA3", "ANXA4", "ANXA5", "ANXA6", "ANXA7",
        "CALM1", "CALM2", "CALM3", "CALU", "CALR", "CANX",
        "CCT2", "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8",
        "TCP1", "PPIA", "PPIB", "PPIC", "FKBP1A", "FKBP4", "FKBP5",
        "VCP", "PSMC1", "PSMC2", "PSMC3", "PSMC4", "PSMC5", "PSMC6",
        "PSMD1", "PSMD2", "PSMD3", "PSMD4", "PSMD5", "PSMD6", "PSMD7",
        "RAB5A", "RAB7A", "RAB11A", "RAB1A", "RAB2A", "RAB3A",
        "RAB4A", "RAB6A", "RAB8A", "RAB10", "RAB14", "RAB18",
        "ARF1", "ARF3", "ARF4", "ARF5", "ARF6",
        "GNAI1", "GNAI2", "GNAI3", "GNAQ", "GNA11", "GNA12", "GNA13",
        "GNAS", "GNB1", "GNB2", "GNB3", "GNB4", "GNG2", "GNG5", "GNG10", "GNG12",
        "YWHAQ", "YWHAZ", "YWHAB", "YWHAE", "YWHAG", "YWHAH",
        "SFN", "RACK1", "GNB2L1",
        "ATP5A1", "ATP5B", "ATP5C1", "ATP5D", "ATP5E", "ATP5F1", "ATP5H", "ATP5I",
        "ATP5J", "ATP5J2", "ATP5L", "ATP5O", "ATP6V0C", "ATP6V1A", "ATP6V1B2", "ATP6V1E1",
        "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFA5", "NDUFA6", "NDUFA7",
        "NDUFB1", "NDUFB2", "NDUFB3", "NDUFB4", "NDUFB5", "NDUFB6", "NDUFB7",
        "NDUFC1", "NDUFC2", "NDUFS1", "NDUFS2", "NDUFS3", "NDUFS4", "NDUFS5", "NDUFS6",
        "NDUFV1", "NDUFV2", "NDUFV3",
        "COX1", "COX2", "COX3", "COX4I1", "COX5A", "COX5B", "COX6A1", "COX6B1", "COX6C",
        "COX7A1", "COX7A2", "COX7B", "COX7C", "COX8A",
        "UQCRB", "UQCRC1", "UQCRC2", "UQCRFS1", "UQCRH", "UQCRQ",
        "SDHA", "SDHB", "SDHC", "SDHD",
        "CYCS", "CYC1", "CMPK1", "CMPK2",
        "MRPS12", "MRPS18A", "MRPS18B", "MRPS21", "MRPS22", "MRPS31",
        "MRPL1", "MRPL2", "MRPL3", "MRPL4", "MRPL9", "MRPL11", "MRPL12", "MRPL13",
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_genes = []
    for g in common_genes:
        if g not in seen:
            seen.add(g)
            unique_genes.append(g)
    
    # Build token-to-index mapping: special tokens first, then genes
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    all_tokens = special_tokens + unique_genes
    token2idx = {token: idx for idx, token in enumerate(all_tokens)}
    
    # Write as JSON (GeneVocab.from_file can load this format)
    with open(vocab_file, "w") as f:
        json.dump(token2idx, f, indent=2)
    
    print(f"Created vocab with {len(token2idx)} tokens at {vocab_file}")
    return token2idx


def create_args_json(save_dir: Path):
    """Create model args.json with default scGPT_human architecture."""
    args = {
        "embsize": 512,
        "nheads": 8,
        "d_hid": 512,
        "nlayers": 12,
        "n_layers_cls": 3,
        "vocab_size": 2463,  # Will be updated after vocab creation
        "dropout": 0.2,
        "pad_token": "<pad>",
        "pad_value": -2,
        "n_input_bins": 51,
    }
    
    args_file = save_dir / "args.json"
    with open(args_file, "w") as f:
        json.dump(args, f, indent=2)
    print(f"Created args.json at {args_file}")
    return args


def create_fake_pretrained(save_dir: Path, args: dict, vocab):
    """
    Create an initialized (random) model and save it as best_model.pt.
    This is NOT a real pretrained model - it provides random initialization
    so the model architecture can be built during training.
    """
    import sys
    sys.path.insert(0, str(save_dir.parent.parent))
    
    from scgpt.model import TransformerModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model with the same architecture
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args["embsize"],
        nhead=args["nheads"],
        d_hid=args["d_hid"],
        nlayers=args["nlayers"],
        nlayers_cls=args.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=args.get("dropout", 0.2),
        pad_token="<pad>",
        pad_value=-2,
        do_mvc=True,
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=10,
        domain_spec_batchnorm=True,
        input_emb_style="category",
        n_input_bins=51,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.8,
        explicit_zero_prob=True,
        use_fast_transformer=True,
        pre_norm=False,
    ).to(device)
    
    # Save state dict
    model_path = save_dir / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Created initialized model weights at {model_path}")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def main():
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "save" / "scGPT_human"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing checkpoint at: {save_dir}")
    
    # Step 1: Create vocab
    vocab = create_vocab_json(save_dir)
    
    # Step 2: Create args.json
    args = create_args_json(save_dir)
    
    # Update args with actual vocab size
    args["vocab_size"] = len(vocab)
    with open(save_dir / "args.json", "w") as f:
        json.dump(args, f, indent=2)
    
    # Step 3: Create initialized model weights
    create_fake_pretrained(save_dir, args, vocab)
    
    print(f"\nCheckpoint initialization complete at: {save_dir}")
    print(f"  - args.json: {os.path.getsize(save_dir / 'args.json')} bytes")
    print(f"  - vocab.json: {os.path.getsize(save_dir / 'vocab.json')} bytes")
    print(f"  - best_model.pt: {os.path.getsize(save_dir / 'best_model.pt') / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()