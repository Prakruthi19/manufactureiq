"""
ManufactureIQ — Self-supervised pre-training on the unlabelled wafers
======================================================================
Input : data/processed/wm811k_unlabelled_maps48.npz   (scripts/download_unlabelled.py)
        data/processed/wm811k_features.npz + wm811k_maps48.npz (for the split + probe)
Output: models/wafer_cnn_encoder_simclr.pt   (conv layers only, to pass to
        train_wafer_cnn.py --init), models/wafer_cnn_encoder_simclr_metrics.json

Why
    Only 26k of the 173k labelled wafers are defective, so the CNN learns its
    filters from little data. WM-811K also has ~638k wafers nobody labelled.
    Pre-training teaches the same conv layers what wafer-map patterns look like
    from those, with no labels; fine-tuning then only has to name the patterns.

How (SimCLR, Chen et al. 2020)
    Each wafer is shown twice with different random changes that don't change
    what the defect is: rotated / mirrored (a ring is a ring whichever way the
    wafer is turned), 15% of failing cells forgotten, a few random failing cells
    added inside the wafer circle. The model is trained so the two versions of
    a wafer land close together and far from the other wafers in the batch
    (NT-Xent loss). No shifting or cropping: position matters (Edge-Loc vs Loc).

No leakage
    Unlabelled wafers from a validation or test lot-block are dropped, so the
    test lots stay unseen in every form, pictures included.

Monitoring (labels used to watch, never to train or choose)
    Every few epochs: k-nearest-neighbour macro-F1 on the validation lots, using
    the frozen embedding and the labelled training wafers. If it rises, the
    model is learning defect structure without having seen a label.

Needs a GPU (Colab T4: ~1 min per epoch). notebooks/wafer_cnn_colab.ipynb runs it.
    python scripts/pretrain_wafer_cnn.py --epochs 30
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from scripts.download_unlabelled import OUT as UNLABELLED                  # noqa: E402
from scripts.train_wafer_classifier import FEATURES, MODELS_DIR, SEED, lot_split  # noqa: E402
from scripts.train_wafer_cnn import MAPS, Maps, WaferCNN                   # noqa: E402

OUT = MODELS_DIR / "wafer_cnn_encoder_simclr.pt"


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WaferCNN(1).features                     # same conv layers as the classifier
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projector = nn.Sequential(nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Linear(256, 64))

    def embed(self, x):
        return self.pool(self.encoder(x))

    def forward(self, x):
        return self.projector(self.embed(x))


def make_view(x: torch.Tensor, circle: torch.Tensor) -> torch.Tensor:
    """Per-wafer random rotation/mirror, forget 15% of failing cells, add ~0.5% noise inside the wafer."""
    k = torch.randint(0, 8, (len(x),), device=x.device)
    out = torch.empty_like(x)
    for t in range(8):
        m = k == t
        if m.any():
            v = x[m].flip(3) if t >= 4 else x[m]
            out[m] = torch.rot90(v, t % 4, (2, 3))
    keep = torch.rand_like(out) > 0.15
    salt = (torch.rand_like(out) < 0.005) & circle
    return torch.clamp(out * keep + salt, max=1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    n = len(z1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def embed_all(model, maps, idx, batch=4096):
    model.eval()
    return torch.cat([F.normalize(model.embed(maps(idx[i:i + batch])), dim=1)
                      for i in range(0, len(idx), batch)])


@torch.no_grad()
def knn_probe(model, maps, y, tr_idx, val_idx, n_classes, k=20):
    """Macro-F1 of a cosine k-NN vote over the frozen embedding (labels only for monitoring)."""
    e_tr, e_val = embed_all(model, maps, tr_idx), embed_all(model, maps, val_idx)
    y_tr = torch.as_tensor(y[tr_idx], device=e_tr.device)
    pred = []
    for i in range(0, len(e_val), 1024):
        nn_idx = (e_val[i:i + 1024] @ e_tr.T).topk(k, dim=1).indices
        pred.append(F.one_hot(y_tr[nn_idx], n_classes).sum(1).argmax(1))
    return float(f1_score(y[val_idx], torch.cat(pred).cpu().numpy(), average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tau", type=float, default=0.2, help="NT-Xent temperature")
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--unlabelled", type=Path, default=UNLABELLED)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU — this will take many hours. In Colab: Runtime → Change runtime type → T4 GPU.")

    print("=" * 66)
    print(f"Self-supervised pre-training (SimCLR) on unlabelled WM-811K ({device})")
    print("=" * 66)
    f, lab = np.load(FEATURES, allow_pickle=False), np.load(MAPS, allow_pickle=False)
    le = LabelEncoder().fit(f["y"])
    y, groups, n_classes = le.transform(f["y"]), f["group"], len(le.classes_)
    tr_idx, val_idx, test_idx, _ = lot_split(y, groups)
    held_out = np.union1d(groups[val_idx], groups[test_idx])

    u = np.load(args.unlabelled, allow_pickle=False)
    size = int(u["size"])
    use = ~np.isin(u["group"], held_out)
    print(f"unlabelled wafers: {len(use):,}; {int((~use).sum()):,} dropped (validation/test lot-blocks) "
          f"→ {int(use.sum()):,} used")
    pool = Maps(u["bits"][use], size, device)
    n = int(use.sum())
    labelled = Maps(lab["bits"], int(lab["size"]), device)

    r = (torch.arange(size, device=device) - (size - 1) / 2) / (size / 2)
    circle = (r[None, :] ** 2 + r[:, None] ** 2 <= 1)[None, None]

    model = SimCLR().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = n // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    probe0 = knn_probe(model, labelled, y, tr_idx, val_idx, n_classes)
    print(f"k-NN probe before training (random conv layers): validation macro-F1 {probe0:.3f}")
    print(f"\n      {'epoch':<7}{'loss':>8}{'kNN val F1':>12}{'min':>6}")
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = np.random.permutation(n)
        total = 0.0
        for s in range(steps_per_epoch):
            x = pool(order[s * args.batch:(s + 1) * args.batch]).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = nt_xent(model(make_view(x, circle)), model(make_view(x, circle)), args.tau)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        row = {"epoch": epoch, "loss": round(total / steps_per_epoch, 4)}
        if epoch % args.probe_every == 0 or epoch == args.epochs:
            row["knn_val_macro_f1"] = round(knn_probe(model, labelled, y, tr_idx, val_idx, n_classes), 4)
        history.append(row)
        probe = f"{row['knn_val_macro_f1']:.3f}" if "knn_val_macro_f1" in row else ""
        print(f"      {epoch:<7}{row['loss']:>8.4f}{probe:>12}{(time.time() - t0) / 60:>6.1f}", flush=True)

    torch.save({"encoder": model.encoder.state_dict(), "size": size}, OUT)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "SimCLR (NT-Xent), rotation/mirror + forget 15% of failing cells + 0.5% in-wafer noise",
        "device": device,
        "unlabelled_wafers_used": n,
        "unlabelled_wafers_dropped_heldout_lots": int((~use).sum()),
        "epochs": args.epochs, "batch": args.batch, "tau": args.tau,
        "knn_probe_before": round(probe0, 4),
        "history": history,
        "note": "k-NN probe uses validation labels for monitoring only; the last epoch is saved.",
    }
    (MODELS_DIR / "wafer_cnn_encoder_simclr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT.relative_to(BASE_DIR)} — fine-tune with:\n"
          f"  python scripts/train_wafer_cnn.py --init {OUT.relative_to(BASE_DIR).as_posix()} --tag pretrained")


if __name__ == "__main__":
    main()
