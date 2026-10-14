"""
ManufactureIQ — CNN comparison for the wafer-map Defect Classifier
===================================================================
Input : data/processed/wm811k_maps48.npz + wm811k_features.npz
        (both from scripts/build_wafer_features.py)
Output: models/wafer_cnn.pt, models/wafer_cnn_metrics.json

The question this answers
    "Why hand-built features + XGBoost and not a CNN on the wafer image?"
    Instead of arguing, train a CNN on the raw failure maps and score it on
    exactly the same wafers, with exactly the same metrics.

Fairness rules
    • Same lot-aware train / validation / test split (lot_split, imported from
      train_wafer_classifier.py) — no lot-block on both sides.
    • Same metrics (scores(), also imported): macro-F1, defect recall,
      false-alarm rate. Accuracy is printed but never used to choose.
    • Epoch chosen on VALIDATION macro-F1; the test lots are scored once.
    • Two comparisons are reported:
        validation — both models trained on the train split only (apples to apples)
        test       — XGBoost was refit on train+validation, the CNN was not
                     (it needs validation for choosing the epoch), so the
                     test comparison slightly favours XGBoost.

Input
    48×48 failure map per wafer (1 = some die in that cell failed). Every
    wafer is resized to the same 48×48 circle, so the CNN can't see the
    product's size either. See build_wafer_features.py.

Imbalance
    • Each epoch uses every defective training wafer but a fresh random 25% of
      the "none" wafers (85% of the data). Over 15 epochs the model still sees
      all of them, and an epoch costs a third of the time.
    • Cross-entropy weighted by 1/sqrt(class frequency) of what an epoch
      actually contains — the softer weighting that did well for XGBoost.
    • Flips / 90° rotations as augmentation: a defect pattern means the same
      thing whichever way the wafer is turned.
    • Prior correction: training saw "none" at 25% and with down-weighted
      loss, so the CNN under-rates how common good wafers are and raises too
      many false alarms. Its "none" probability is multiplied by a factor
      chosen on VALIDATION macro-F1 (like XGBoost's review threshold).

Needs PyTorch (pip install torch). It isn't in requirements.txt: the deployed
app doesn't use the CNN, and torch would triple the Docker image.

Pre-training (optional)
    --init models/wafer_cnn_encoder_simclr.pt starts the conv layers from the
    self-supervised model (scripts/pretrain_wafer_cnn.py) instead of random
    weights. --label-fraction 0.1 keeps only 10% of the training lot-blocks,
    to measure how much pre-training helps when labels are scarce. --tag names
    the output files so runs don't overwrite each other.

Runs on a laptop CPU (~2-3 min per epoch) or a GPU, e.g. free Colab
(notebooks/wafer_cnn_colab.ipynb):
    python scripts/train_wafer_cnn.py              # default 15 epochs
    python scripts/train_wafer_cnn.py --epochs 3   # quick check
    python scripts/train_wafer_cnn.py --init models/wafer_cnn_encoder_simclr.pt --tag pretrained
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from scripts.train_wafer_classifier import FEATURES, MODELS_DIR, SEED, lot_split, scores  # noqa: E402

MAPS = BASE_DIR / "data" / "processed" / "wm811k_maps48.npz"
NONE_SCALES = [1, 1.5, 2, 3, 4, 6, 8, 12, 16]


class WaferCNN(nn.Module):
    """Four conv blocks (48→24→12→6→3), global average pool, one linear layer.
    ~0.28M parameters: small on purpose — 26k defective wafers can't feed a big
    net, and it must train on a laptop CPU. One conv in the two high-resolution
    blocks (where convs are expensive), two in the low-resolution ones."""

    def __init__(self, n_classes: int):
        super().__init__()

        def block(c_in, c_out, n_conv):
            layers = []
            for i in range(n_conv):
                layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False),
                           nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers, nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16, 1), block(16, 32, 1), block(32, 64, 2), block(64, 128, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def unpack(bits: np.ndarray, size: int) -> torch.Tensor:
    """Bit-packed rows → float tensor (N, 1, size, size)."""
    m = np.unpackbits(bits, axis=1)[:, : size * size]
    return torch.from_numpy(m.reshape(-1, 1, size, size).astype(np.float32))


class Maps:
    """Failure maps served by row index. On a GPU every map is unpacked once into
    GPU memory as uint8 (~0.4 GB for the labelled set), so batches never wait on
    the CPU; on a CPU they stay bit-packed and are unpacked per batch."""

    def __init__(self, bits: np.ndarray, size: int, device: str):
        self.bits, self.size, self.device = bits, size, device
        self.gpu = None
        if device == "cuda":
            m = np.unpackbits(bits, axis=1)[:, : size * size].reshape(-1, 1, size, size)
            self.gpu = torch.from_numpy(m).to(device)

    def __call__(self, idx: np.ndarray) -> torch.Tensor:
        if self.gpu is not None:
            return self.gpu[torch.as_tensor(idx, device=self.device)].float()
        return unpack(self.bits[idx], self.size)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random flips and 90° rotations, one draw per batch (batches are shuffled)."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


@torch.no_grad()
def predict_proba(model, maps, idx, device, batch=1024):
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        x = maps(idx[i:i + batch]).to(device)
        out.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--none-share", type=float, default=0.25, help="share of 'none' wafers used per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (ignored on GPU)")
    ap.add_argument("--init", type=Path, help="pre-trained encoder (scripts/pretrain_wafer_cnn.py)")
    ap.add_argument("--label-fraction", type=float, default=1.0,
                    help="keep this share of the training lot-blocks (labels-scarce experiment)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. 'pretrained'")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(args.threads)

    print("=" * 66)
    print(f"Defect Classifier — CNN comparison on WM-811K ({device})")
    print("=" * 66)
    f = np.load(FEATURES, allow_pickle=False)
    m = np.load(MAPS, allow_pickle=False)
    if not np.array_equal(f["sample_id"], m["sample_id"]):
        sys.exit("Feature table and map file are out of sync — re-run scripts/build_wafer_features.py")
    bits, size, groups = m["bits"], int(m["size"]), f["group"]
    le = LabelEncoder().fit(f["y"])
    y = le.transform(f["y"])
    classes, n_classes = list(le.classes_), len(le.classes_)
    none_id = classes.index("none")

    tr_idx, val_idx, test_idx, _ = lot_split(y, groups)
    if args.label_fraction < 1:
        blocks = np.unique(groups[tr_idx])
        keep = np.random.default_rng(SEED).choice(blocks, max(1, round(len(blocks) * args.label_fraction)),
                                                  replace=False)
        tr_idx = tr_idx[np.isin(groups[tr_idx], keep)]
        print(f"label fraction {args.label_fraction:.0%}: {len(keep)} of {len(blocks)} training lot-blocks, "
              f"per class {dict(zip(map(str, classes), np.bincount(y[tr_idx], minlength=n_classes).tolist()))}")
    maps = Maps(bits, size, device)
    print(f"train {len(tr_idx):,} | validation {len(val_idx):,} | test {len(test_idx):,}  "
          f"(same lot-aware split as XGBoost)")

    tr_none, tr_defect = tr_idx[y[tr_idx] == none_id], tr_idx[y[tr_idx] != none_id]
    n_none = int(len(tr_none) * args.none_share)
    counts = np.bincount(y[tr_defect], minlength=n_classes).astype(float)
    counts[none_id] = n_none
    counts = np.maximum(counts, 1)                 # a class can vanish at small label fractions
    weights = (counts.sum() / (n_classes * counts)) ** 0.5
    model = WaferCNN(n_classes).to(device)
    if args.init:
        model.features.load_state_dict(torch.load(args.init, map_location=device)["encoder"])
        print(f"conv layers initialised from {args.init.name} (self-supervised pre-training)")
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {size}×{size} input, "
          f"{args.epochs} epochs, batch {args.batch}")
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil((len(tr_defect) + n_none) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)

    print(f"\n      {'epoch':<7}{'loss':>8}{'val macroF1':>13}{'defect rec':>12}{'false alarm':>13}{'min':>6}")
    best_f1, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = np.random.permutation(np.concatenate([tr_defect, np.random.choice(tr_none, n_none, replace=False)]))
        total = 0.0
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            x = augment(maps(b).to(device))
            t = torch.from_numpy(y[b]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        p_val = predict_proba(model, maps, val_idx, device)
        s = scores(y[val_idx], p_val.argmax(1), none_id, n_classes)
        history.append({"epoch": epoch, "train_loss": round(total / len(order), 4), **s})
        print(f"      {epoch:<7}{total / len(order):>8.4f}{s['macro_f1']:>13.3f}{s['defect_recall']:>12.3f}"
              f"{s['false_alarm_rate']:>13.3f}{(time.time() - t0) / 60:>6.1f}", flush=True)
        if s["macro_f1"] > best_f1:
            best_f1, best_epoch = s["macro_f1"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print(f"      → best epoch by validation macro-F1: {best_epoch}")
    model.load_state_dict(best_state)

    # ── Undo the training mix's bias against "none" (chosen on validation) ──
    p_val = predict_proba(model, maps, val_idx, device)
    val_by_scale = {}
    for k in NONE_SCALES:
        q = p_val.copy()
        q[:, none_id] *= k
        val_by_scale[k] = scores(y[val_idx], q.argmax(1), none_id, n_classes)
    none_scale = max(NONE_SCALES, key=lambda k: val_by_scale[k]["macro_f1"])
    val = val_by_scale[none_scale]
    print(f"      → 'none' probability × {none_scale} (validation macro-F1 "
          f"{val_by_scale[1]['macro_f1']:.3f} → {val['macro_f1']:.3f}, false alarms "
          f"{val_by_scale[1]['false_alarm_rate']:.3f} → {val['false_alarm_rate']:.3f})")

    # ── Test lots, scored once ────────────────────────────────────────────────
    p_te = predict_proba(model, maps, test_idx, device)
    p_te[:, none_id] *= none_scale
    y_te, pred_te = y[test_idx], p_te.argmax(1)
    test = scores(y_te, pred_te, none_id, n_classes)
    print(classification_report(y_te, pred_te, target_names=classes, digits=3, zero_division=0))

    xgb_path = MODELS_DIR / "wafer_classifier_metrics.json"
    xgb = json.loads(xgb_path.read_text(encoding="utf-8")) if xgb_path.exists() else None
    if xgb:
        xv = xgb["strategy_comparison_validation"][xgb["selected_strategy"]]
        xt = xgb["test"]
        print(f"      {'':<26}{'macroF1':>9}{'patterns':>10}{'defect rec':>11}{'false alarm':>12}")
        for label, r in [("XGBoost  validation", xv), ("CNN      validation", val),
                         ("XGBoost  test", xt), ("CNN      test", test)]:
            print(f"      {label:<26}{r['macro_f1']:>9.3f}{r['macro_f1_patterns']:>10.3f}"
                  f"{r['defect_recall']:>11.3f}{r['false_alarm_rate']:>12.3f}")

    suffix = f"_{args.tag}" if args.tag else ""
    prf = precision_recall_fscore_support(y_te, pred_te, labels=range(n_classes), zero_division=0)
    torch.save({"state_dict": model.state_dict(), "classes": [str(c) for c in classes], "size": size,
                "none_scale": none_scale}, MODELS_DIR / f"wafer_cnn{suffix}.pt")
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": f"CNN, 4 conv blocks, {sum(p.numel() for p in model.parameters()):,} parameters",
        "input": f"{size}×{size} failure map (resized; product size hidden)",
        "device": device,
        "init": args.init.name if args.init else "random",
        "label_fraction": args.label_fraction,
        "train_wafers": int(len(tr_idx)),
        "imbalance": f"{args.none_share:.0%} of 'none' wafers resampled each epoch, cross-entropy weighted by "
                     "1/sqrt(class frequency), flip/rotation augmentation",
        "split": "same lot-aware split as the XGBoost classifier (lot_split)",
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "none_scale": none_scale,
        "history": history,
        "validation": val,
        "test": test,
        "test_per_class": {c: {"precision": round(float(prf[0][i]), 3), "recall": round(float(prf[1][i]), 3),
                               "f1": round(float(prf[2][i]), 3), "support": int(prf[3][i])}
                           for i, c in enumerate(classes)},
        "test_confusion_matrix": confusion_matrix(y_te, pred_te, labels=range(n_classes)).tolist(),
        "note": "XGBoost's test model was refit on train+validation; the CNN trained on train only.",
    }
    (MODELS_DIR / f"wafer_cnn{suffix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved models/wafer_cnn{suffix}.pt + wafer_cnn{suffix}_metrics.json")


if __name__ == "__main__":
    main()
