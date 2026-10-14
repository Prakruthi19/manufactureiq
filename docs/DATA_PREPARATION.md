# From raw wafer data to a trained model: the data-preparation story

This is the path one wafer takes, from the downloaded file to the numbers a model
learns from. Every step has a reason, and most of the reasons are about **not
letting the model cheat**. Read this before an interview. It's the part people probe.

---

## Step 1: What the dataset actually contains

**WM-811K** (Wu, Jang & Chen, IEEE Trans. Semiconductor Manufacturing, 2015) is
811,457 wafers from 46,393 production lots of a real fab. It contains **no image
files**. Each wafer is a grid of numbers:

```
0 = no chip here (outside the round wafer)   1 = chip passed   2 = chip failed
```

The original file (`LSWMD.pkl`, ~2 GB) has one row per wafer:

| column | example | used? |
|---|---|---|
| `waferMap` | 26×30 grid of 0/1/2 | **yes**: this is the data |
| `failureType` | Edge-Ring | **yes**: the expert's label (the answer) |
| `lotName`, `waferIndex` | lot1, 1–25 | for splitting only (step 5) |
| `dieSize` | 600 | **no**: identifies the product (step 4) |
| `trianTestLabel` | Training / Test | **no**: the authors' split isn't lot-safe (step 5) |

Only **172,950** wafers have an expert label. Those are the ones we train on:

| label | wafers | |
|---|---|---|
| none | 147,431 | 85%: most wafers are fine |
| Edge-Ring | 9,680 | |
| Edge-Loc | 5,189 | |
| Center | 4,294 | |
| Loc | 3,593 | |
| Scratch | 1,193 | |
| Random | 866 | |
| Donut | 555 | |
| Near-full | 149 | 0.09%: rarest |

The other ~638,000 wafers are unlabelled and unused. They're a future opportunity
(semi-supervised learning).

---

## Step 2: Download and store compactly: `scripts/download_data.py`

The original download link is dead, so we use a sharded copy on Hugging Face.
For each labelled wafer we keep only the raw facts:

```
label, grid height h, grid width w, number of real chips, (x, y) of every failing chip
```

Storing only the *failing* chips (not every 0 and 1) shrinks 2 GB to **37 MB**
(`data/raw/wm811k/`, 2,048 small `.npz` files). The full grid can be rebuilt
exactly, because the wafer is the circle inscribed in the grid.

**Trust check:** the script counts every class and compares it with the counts
published in the paper. They match exactly, so nothing was lost or relabelled.

One real wafer (`sample_id 172032`, label Edge-Ring), drawn as a picture
(`.` pass, `X` fail):

```
                      . X . . X . . .
                . . . . . . . . . . . . . .
            X . . . . . . . X . . . . . . . . .
          . . . . . . . . . . . . . . . . . . . X
        . . . . . . . . . . . . . . . . . . . . . X
      . . . . . . . . . . . . . . . . . . . . . . . X
    . . . . . . . . . . . . . . . X . . . . . . . . . X
  . . . . . . . . . . . . . . . . . . . . . . . . . . X .
  X . X . . . . . . . . . . . . . . . . . . . . . . . . X
  . . . . . . . . . . . . . . . . . . . X . . . . . . . X
. . . . . . . . . . . . . . X . . . . . . . . . . . . . X .
. . . . . . . . . . X . . . . . . . . . . . . . . . . . X X
. . . . . . . . . . . . . . . X . . . . . . . . X . . . . X
. . . . . X . . . . . . . . . . . . . . . . . . . . . . X X
. . . . . . . . . . . . . . . X X . . . . . . . . . . . . X
. . . . . . . . . . . . . . . . . . . . . . . . . . . . X .
  . . . X . . . . . . . . . . . . . . . . . . . . . . . X
  . . . . . . . X . . . . . . . . . . . . . . . . . X . X
  . . . . . . . . . . . . . . . . . . . X . . . . . . X .
    . . X . . . . . . . . . . . . . . . . . . X . . X X
      . . . . . . . . . . . . . . . . . . . . . . X X
        . . . . . . . . . . . . . . . . . . . . . X
          . . . X . . . . . . . . . . . . . . X X
            . . . . X . . . . . . . . . . . X .
                X . . . . . . . . X . . X .
                      X X X X X X X .
```
The failures trace the rim, so an expert says Edge-Ring.

---

## Step 3: Turn the grid into meaning: `core/wafer_features.py`

A model can't look at a wafer the way an engineer does, so we compute what an
engineer would look for: **26 numbers per wafer**, each with a physical meaning.

| group | example feature | the question it answers | pattern it reveals |
|---|---|---|---|
| amount | `fail_density` | what share of chips failed? | Near-full, Random |
| where (rings) | `ring0_density` … `ring5_density` | failure rate in 6 rings, centre → edge | Center, Donut, Edge-Ring |
| where (relative) | `center_vs_wafer`, `edge_vs_wafer` | is the centre / edge worse than the wafer's average? | Center, Edge-* |
| around the edge | `edge_sector_entropy` | do edge failures go all the way round, or sit in one spot? | Edge-Ring vs Edge-Loc |
| clusters | `largest_cluster_frac`, `_elongation`, `_fill` | are failures clumped? into what shape? | Loc, Scratch |
| lines | `line_score` | do failures line up? | Scratch |

The same wafer as features:

| feature | value | reads as |
|---|---|---|
| fail_density | 0.098 | ~10% of chips failed |
| edge_vs_wafer | **3.28** | edge fails 3.3× the average → edge pattern |
| edge_sector_entropy | 0.871 | spread round the rim → **ring**, not local |

**One function serves both training and the live app** (`features_from_points` and
`features_from_map` share the code, and a test checks they agree), so the model
never sees features computed differently in production.

---

## Step 4: Hide the product from the model (and a leak I found and fixed)

**The risk:** wafer grids come in 346 sizes, which is roughly one per product. Some
products produce mostly one pattern. If the model can tell which product a wafer
is, it can learn *"wafers with ~533 chips are usually Edge-Ring"*. That's a shortcut
that scores well on the test set (same products) and fails on a new product.

**The rule:** never give the model the size. Every feature is a **ratio** (failing ÷
total, centre rate ÷ wafer rate) or a **position on a 0–1 scale** (0 = centre,
1 = edge). "30% failed, mostly near the edge" reads the same on a small wafer and
a big one.

**The leak:** the first version had one more feature, `log_n_fail`, the raw
*count* of failing chips. Alone it looks harmless. But together with
`fail_density`:

```
count ÷ failure rate = total chips      e.g. 154 ÷ 0.289 ≈ 533 chips  → the product
```

and the trained model was using it: `log_n_fail` appeared repeatedly in its
decision trees. It was removed and the model retrained. *(Results before → after:
see step 8.)*

**Interview line:** "I excluded product size on purpose, then audited my own
features and found one that leaked it indirectly. I removed it and measured the
cost."

---

## Step 5: Split by lot, not by wafer: `lot_split()` in `train_wafer_classifier.py`

Wafers from the same lot went through the same machines at the same hour, so they
look alike. With a random split, a test wafer's lot-mates sit in training, and the
model is graded on near-copies of what it studied.

So wafers are grouped into **blocks of 1,000 consecutive wafers (~40 lots)**, and
whole blocks go to one side:

| set | wafers | used for |
|---|---|---|
| train | 110,950 | learning |
| validation | 28,000 | comparing options, picking thresholds |
| test | 34,000 | **scored once, at the very end** |

The split keeps each class's share similar on every side (stratified), and the
code asserts that no block appears twice. The CNN imports the **same function**,
so both models are graded on identical wafers.

---

## Step 6: Deal with the imbalance, and measure it honestly

85% of wafers are "none". A model that always says "none" gets 85% accuracy
and is useless. So:

- **Accuracy is never used to choose anything.** The main score is **macro-F1**:
  the F1 of each of the 9 classes, averaged, so Near-full (149 wafers) counts as
  much as none (147,431).
- Also reported: **defect recall** (of the defective wafers, how many did we flag?)
  and **false-alarm rate** (of good wafers, how many did we wrongly flag?).
- Six ways of handling the imbalance were compared on the validation set (no
  balancing, class weights, softer weights, under-sampling "none", SMOTE,
  Random Forest). The winner was picked by validation macro-F1.
- SMOTE is applied **inside the training data only**. Applying it before
  splitting leaks synthetic copies of test wafers into training.

---

## Step 7: Train, and let the data set the rules

XGBoost builds 400 rounds × 9 classes = **3,600 small trees** (max 6 questions
deep). Each question is a cut-off such as `center_vs_wafer < 2.666`. **Nobody
writes those numbers.** Training tries thousands of cut-offs on every feature and
keeps the ones that best separate the expert labels. For example, on defective
wafers:

| question | Center wafers caught | how clean the "yes" side is |
|---|---|---|
| centre > 1.5× average | 100% | 66% Center (lots of Loc mixed in) |
| centre > 3.0× average | 53% | **95% Center** |
| centre > 5.0× average | 9% | 98% Center |

What *I* chose: the features, the tree size, the number of rounds, and the
human-review rule (wafers the model is unsure about go to an engineer).

---

## Step 8: Results, and the CNN comparison

All numbers below are on the **34,000 test wafers from lots never seen in training**,
scored once.

**Removing the leak (step 4)**

| | with `log_n_fail` | without (current) |
|---|---|---|
| Validation macro-F1 | 0.889 | 0.886 |
| **Test macro-F1** | 0.874 | **0.861** |
| Defective wafers caught | 91.9% | 90.8% |
| False alarms on good wafers | 0.6% | 0.6% |

The score dropped by 0.013. That gap is roughly how much the old model had been
leaning on the product shortcut. The lower number is the honest one.

**CNN on the raw maps vs XGBoost on features**

The CNN gets the 48×48 failure map (the grid from step 2, resized). Resizing every
wafer to the same size hides the product from the CNN too. Both models use the same
split (step 5) and the same scores (step 6).

| | macro-F1 | defects caught | false alarms |
|---|---|---|---|
| XGBoost, validation | **0.886** | 93.7% | **0.7%** |
| CNN, validation | 0.847 | 94.3% | 1.1% |
| XGBoost, test | **0.861** | 90.8% | **0.6%** |
| CNN, test | 0.840 | **92.3%** | 1.2% |

- XGBoost wins on macro-F1 and raises half the false alarms. The CNN catches a few
  more defects, but by flagging more good wafers too.
- The CNN needed its own imbalance handling: a fresh 25% of "none" each epoch plus
  class weights. That made it over-predict defects, so its "none" probability was
  scaled up by a factor chosen on validation (×6), the same way XGBoost's review
  threshold is chosen on validation.
- **Fair caveats:** the CNN is small (283k parameters, 15 epochs on a laptop CPU),
  and XGBoost's final model was refit on train+validation while the CNN's wasn't. The
  validation rows compare like with like, and XGBoost still wins there.

**Decision:** keep XGBoost on features. It scores higher, raises fewer false alarms
and can explain every call. The CNN stays in the repo as evidence
(`scripts/train_wafer_cnn.py`, `models/wafer_cnn_metrics.json`).

---

## Step 9: Make the confidence honest, then watch the inputs

Two things happen after training, both inside `scripts/train_wafer_classifier.py`.

**Calibration** (`core/calibration.py`). SMOTE and boosting make the model sound
surer than it is: test wafers it called 86% certain were right only 73% of the time.
One number, the temperature **T = 1.45**, is fitted on the validation lots and softens
every probability. It never changes which pattern wins, so every score in step 8 is
unchanged. Only the confidence moves:

| test wafers in the 0.8–0.9 confidence bucket | confidence | actually correct |
|---|---|---|
| before | 86% | 73% |
| after | 86% | **86%** |

The human-review threshold is then chosen on the calibrated numbers (**0.81**), and the
live agent applies the same T. On test, 4.6% of wafers go to an engineer.

**Drift reference** (`core/drift.py`). Each of the 26 features from step 3 is cut into
10 buckets at the training deciles, and the share of training wafers per bucket is
saved to `models/drift_reference.json`. In production every wafer's features are
logged, and the latest 500 are compared with that reference using the Population
Stability Index (below 0.10 stable, above 0.25 major shift). This needs no labels, so a
new product or process change shows up immediately. Two checks at training time show it
works both ways: held-out test lots come out **stable** (max PSI 0.064), and a sample
of only defective wafers sets off a **major** alarm (max PSI 3.4), as it should.

---

## Step 10 (in progress): Learn from the unlabelled wafers

Only 172,950 of the 811,457 wafers have labels. `scripts/download_unlabelled.py` keeps
the 48×48 failure map of the other ~638k (empty maps dropped), and
`scripts/pretrain_wafer_cnn.py` pre-trains the CNN's layers on them with SimCLR: it
learns that a rotated, mirrored or slightly noised copy of a wafer is still the same
wafer. Wafers from validation or test lot-blocks are dropped first, so the split from
step 5 still holds. `scripts/train_wafer_cnn.py --init` then fine-tunes on the labels.
It needs a GPU and runs in `notebooks/wafer_cnn_colab.ipynb`. **No results yet.**

---

## The whole pipeline, in order

```
python scripts/download_data.py          # raw data, checked against published counts
python scripts/build_wafer_features.py   # 26 features + 48×48 CNN maps, every labelled wafer
python scripts/train_models.py           # SECOM monitor, XGBoost classifier (+ calibration, drift reference), similarity index, demo wafers
python scripts/train_wafer_cnn.py        # optional: CNN comparison on the same split
# GPU only (notebooks/wafer_cnn_colab.ipynb):
python scripts/download_unlabelled.py    # ~638k unlabelled wafer maps
python scripts/pretrain_wafer_cnn.py     # SimCLR pre-training, then train_wafer_cnn.py --init …
```
