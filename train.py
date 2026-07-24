"""
Self-Supervised Training & Linear Probing (`train.py`)

Trains the PatchTST masked autoencoder on ZTF light curves (real Parquet or
the built-in mock survey), then evaluates representation quality by fitting a
logistic-regression probe on frozen [CLS] embeddings against the validation
ground truth (mock mode only - real ZTF data has no debris labels).

Examples:
    python train.py --use_mock True  --epochs 10 --batch_size 64
    python train.py --use_mock False --data_dir data/ztf_parquet --limit 20000
"""
import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.mock_generator import generate_mock_ztf_dataset
from src.data.ztf_dataset import LightCurveDataset, PolarsZTFDataset, collate_lightcurves
from src.models.patchtst_mae import PatchTSTMaskedAutoencoder

SEQ_LEN = 256


def str2bool(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes")


@torch.no_grad()
def _extract_latents(model, loader, device):
    """Frozen [CLS] embeddings and labels for every object in the loader."""
    latents, labels = [], []
    for batch in loader:
        x = batch["x"].to(device)
        t = batch.get("t")
        t = t.to(device) if t is not None else None
        latents.append(model.extract_cls_latent(x, t).cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.vstack(latents), np.concatenate(labels)


def run_linear_probing(model, train_loader, val_loader, device) -> float:
    """
    Linear separability of frozen [CLS] latents (clean vs injected debris).
    This is a *validation* protocol: labels train only the tiny linear probe,
    never the encoder.
    """
    model.eval().to(device)
    X_train, y_train = _extract_latents(model, train_loader, device)
    X_val, y_val = _extract_latents(model, val_loader, device)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("[!] Skipping linear probing: both classes must be present in train and val.")
        return 0.5

    scaler = StandardScaler().fit(X_train)
    probe = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    probe.fit(scaler.transform(X_train), y_train)

    val_probs = probe.predict_proba(scaler.transform(X_val))[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)

    print("\n" + "=" * 60)
    print("[+] LINEAR PROBING ON FROZEN [CLS] LATENTS")
    print("=" * 60)
    print(f"  * Validation ROC-AUC : {auc:.4f}{'  [>0.95 target met]' if auc >= 0.95 else ''}")
    print(f"  * Validation PR-AUC  : {pr_auc:.4f}")
    print(f"  * Train / Val sizes  : {len(y_train)} / {len(y_val)}  "
          f"(debris fraction: {y_train.mean():.2%} / {y_val.mean():.2%})")
    print("=" * 60 + "\n")
    return float(auc)


def load_records(args):
    """Returns the list of per-object records for the chosen data source."""
    real_path = args.parquet or (None if str2bool(args.use_mock) else args.data_dir)
    if real_path is None:
        return generate_mock_ztf_dataset(
            n_clean=args.n_clean, n_debris=args.n_debris, seq_len=SEQ_LEN, seed=args.seed
        )
    if not os.path.exists(real_path):
        raise FileNotFoundError(
            f"Light-curve archive not found at {real_path!r}. "
            "Pass a .parquet/.csv file or a directory of them, "
            "or run with `--use_mock True`."
        )
    print(f"  * Streaming real archive: {real_path} (limit={args.limit})")
    ds = PolarsZTFDataset(real_path, seq_len=SEQ_LEN)
    records = ds.load_records(limit=args.limit)
    if not records:
        raise RuntimeError("No light curves passed the minimum-observation filter.")
    print(f"  * Loaded {len(records)} objects from the real archive.")
    return records


def main():
    parser = argparse.ArgumentParser(description="Dark Satellite Hunter | PatchTST-MAE Training")
    parser.add_argument("--use_mock", type=str, default="True", help="Use built-in mock survey (True/False)")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Real light-curve archive (.parquet/.csv file or directory); overrides --use_mock")
    parser.add_argument("--data_dir", type=str, default="data/ztf_parquet", help="Path to ZTF Parquet files")
    parser.add_argument("--limit", type=int, default=20000, help="Max objects to load from Parquet")
    parser.add_argument("--n_clean", type=int, default=500, help="Mock: number of clean objects")
    parser.add_argument("--n_debris", type=int, default=120, help="Mock: number of debris injections")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb", type=str, default="False", help="Log to Weights & Biases (True/False)")
    parser.add_argument("--fast_dev_run", type=str, default="False", help="Single-step dry run (True/False)")
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    print("\n[*] Dark Satellite Hunter - self-supervised training")
    source = args.parquet or ("mock survey" if str2bool(args.use_mock) else args.data_dir)
    print(f"  * Data source : {source}")
    print(f"  * Mask ratio  : {args.mask_ratio:.0%} | Epochs: {args.epochs} | Batch: {args.batch_size}\n")

    # ---- Data: stratified split so both classes appear on each side -------
    records = load_records(args)
    labels = np.array([int(r.get("label", 0)) for r in records])
    stratify = labels if len(np.unique(labels)) > 1 else None
    idx_train, idx_val = train_test_split(
        np.arange(len(records)), test_size=0.2, random_state=args.seed, stratify=stratify
    )
    train_ds = LightCurveDataset([records[i] for i in idx_train], seq_len=SEQ_LEN, train=True, seed=args.seed)
    val_ds = LightCurveDataset([records[i] for i in idx_val], seq_len=SEQ_LEN, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_lightcurves, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_lightcurves, num_workers=args.num_workers)

    # ---- Model -------------------------------------------------------------
    model = PatchTSTMaskedAutoencoder(
        in_channels=2, seq_len=SEQ_LEN, patch_len=16,
        d_model=256, enc_layers=4, enc_heads=8,
        dec_dim=128, dec_layers=2, dec_heads=4,
        mask_ratio=args.mask_ratio, lr=args.lr, max_epochs=args.epochs,
    )

    # ---- Trainer -------------------------------------------------------------
    loggers = [CSVLogger("logs", name="patchtst_mae")]
    if str2bool(args.wandb):
        try:
            loggers.append(WandbLogger(project="dark-satellite-hunter", log_model=True))
        except Exception as exc:
            print(f"[!] W&B logger unavailable ({exc}); continuing with CSV logging.")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
        filename="patchtst-mae-{epoch:02d}-{val_loss:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=loggers,
        callbacks=[checkpoint_cb],
        accelerator="auto",
        devices=1,
        fast_dev_run=str2bool(args.fast_dev_run),
        log_every_n_steps=5,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    # ---- Evaluation ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_linear_probing(model, train_loader, val_loader, device=device)

    if checkpoint_cb.best_model_path:
        print(f"[+] Best checkpoint: {checkpoint_cb.best_model_path}")
        print("    Use it with:  python run_discovery.py --ckpt_path <path-above>")
    print("[+] Training complete. Logs in `logs/patchtst_mae/`.")


if __name__ == "__main__":
    main()
