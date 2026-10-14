# ManufactureIQ — Interview Guide

Why every decision in this project was made, what the numbers really mean,
and how to answer the questions you'll get. Read this before you present it.

---

## 1. The 30-second pitch

> ManufactureIQ is a multi-agent quality-control system for semiconductor fabs.
> A wafer comes off the line; seven agents validate it, classify its failure
> pattern, explain the likely cause, check whether the problem is repeating
> across a lot, a tool or a material lot, alert the supplier if the evidence
> says it's their material, and write an ISO 9001 corrective-action report.
>
> The two ML models are trained on **real fab data**: 172,950 expert-labelled
> wafer maps (WM-811K) and 1,567 production runs with 590 sensors (SECOM).
> The defect classifier reaches **macro-F1 0.861** on lots it never saw and
> catches **91% of defective wafers** with a **0.6% false-alarm rate**, and it
> beat a CNN trained on the raw wafer maps under the same split.
> Its confidence is calibrated, so "86% sure" really means right 86% of
> the time, and a drift monitor flags when live wafers stop looking like
> the training data.
> The process model is weak on future data, and the system is built to
> say so rather than hide it.

---

## 2. Why I replaced the original dataset

The first version used **AI4I 2020**. Three problems an interviewer would find:

| Problem | Why it matters |
|---|---|
| **It's synthetic.** The authors generated it to imitate a milling machine. | "Trained on real data" was not true. |
| **Labels are rules on the features.** e.g. power outside 3,500–9,000 W ⇒ "power failure". | The 99% accuracy was the model re-learning the generator's if-statements, not discovering anything. |
| **Machine failures were renamed as product defects.** Tool-wear failure → `structural_crack`, power failure → `assembly_misalignment`. | Indefensible. A high-power-draw reading was reported as an assembly defect. |

The old code is kept in `scripts/legacy_ai4i/` and `models/legacy_ai4i/`, so you can
show the evolution: *"I found my first dataset was synthetic and my labels didn't
match the domain, so I rebuilt it on real data."* That's a stronger story than never
having made the mistake.

---

## 3. The two datasets, and what each one does

### WM-811K → Defect Classifier
- **What:** 811,457 wafer maps from 46,393 lots in a real fab (Wu, Jang & Chen,
  *IEEE Trans. Semiconductor Manufacturing*, 2015). 172,950 were labelled by
  engineers into 9 classes: `none, Center, Donut, Edge-Ring, Edge-Loc, Loc,
  Random, Scratch, Near-full`.
- **A wafer map** is a grid: 0 = no die, 1 = die passed electrical test, 2 = failed.
  The *spatial pattern* of failures points to the process step that caused them
  (a ring at the edge → etch/edge effects; a line → mechanical scratch).
- **Imbalance:** `none` is 85%; `Near-full` is 149 wafers (0.09%). 1,000× rarer.

### SECOM → Sensor Parser (SPC) + Trend Detector (process risk)
- **What:** 1,567 runs from a real semiconductor line, 590 anonymised sensor
  signals each, pass/fail at line test, timestamped July–October 2008 (UCI repository).
- **Imbalance:** 104 failures (6.6%).

### Honest caveat — say this before they ask
The two datasets come from **different fabs**. In the demo, a wafer map and a
SECOM run are *paired* to simulate one inspection; they aren't really linked.
Tool, supplier and material-lot IDs are **simulated** too; neither dataset has them.
Each model is trained on the best real data available *for its job*.

### How I know the data is genuine
The WM-811K copy came from a Hugging Face mirror (the original link is dead and
Kaggle needs an account). `scripts/download_data.py` checks the label counts
against the published distribution, and **all 9 classes match exactly**
(147,431 none, 9,680 Edge-Ring, … 149 Near-full). I also didn't use the mirror's
pre-computed features; every feature is computed from the raw failing-die
coordinates by my own code.

---

## 4. Defect Classifier: decisions and why

### 4.1 Features instead of a CNN
`core/wafer_features.py` computes **26 features**, each with a physical meaning:

| Group | Features | Detects |
|---|---|---|
| Density | failure rate | Near-full |
| Radial | failure density in 6 rings; centre-vs-wafer, edge-vs-wafer, middle-vs-(centre & edge) | Center, Edge-Ring, Donut |
| Angular | how evenly edge failures spread round 8 sectors (entropy, CV) | Edge-Loc (one sector) vs Edge-Ring (all round) |
| Clusters | connected groups of failing die: largest cluster share, elongation, fill, isolated-die share | Loc (compact blob), Random (scattered) |
| Lines | max share of failures in one 2-die-wide band over 18 directions | Scratch |

**Why not a CNN?** I tested it instead of arguing it: see §4.6. Short version: on
the same lot-aware split, the feature model scored higher (macro-F1 0.861 vs 0.840),
raised half the false alarms, and every prediction is explainable ("72% of failures
sit in the outer ring, spread evenly" → Edge-Ring), which the root-cause agent needs.

**Deliberately excluded:** wafer size / die count. It identifies the *product*, so the
model could learn "product X usually has pattern Y", which is a shortcut, not the pattern.
Every feature is a ratio or a 0–1 position, so it reads the same on any wafer size.

**A leak I found in my own features, and fixed.** The first version also had
`log_n_fail`, the raw *count* of failing die. Alone it looks harmless, but
count ÷ failure rate = die count, so together with `fail_density` it handed the model
the product size by the back door. The trained trees were using it. I removed it and
retrained (27 → 26 features):

| | with the leak | without |
|---|---|---|
| Validation macro-F1 | 0.889 | 0.886 |
| **Test macro-F1** | 0.874 | **0.861** |
| Defective wafers caught (test) | 91.9% | 90.8% |
| False alarms (test) | 0.6% | 0.6% |

The score dropped a little, meaning part of the old score was the product shortcut.
I report the lower, honest number. **Interview line:** "I audited my own features for
leakage, found one that reconstructed the product size, removed it and measured the
cost: 0.013 macro-F1." The full walk-through is in [DATA_PREPARATION.md](DATA_PREPARATION.md).

**One function, used by training AND serving:** `features_from_points` (training, from
raw coordinates) and `features_from_map` (live, from a grid) call the same code, and a
test asserts they agree. That rules out train/serve skew.

### 4.2 Lot-aware splitting (the leakage control)
Wafers in the same lot go through the same tools at the same time, so they look alike.
I **measured** this: neighbouring defective wafers in the file share a pattern **60.5%**
of the time vs **23.7%** if the order were random. A random split would put lot-mates
in train and test and inflate the score.

So the split is by **lot-block** (1,000 consecutive wafers ≈ 40 lots) using
`StratifiedGroupKFold`: train 110,950 / validation 28,000 / test 34,000, with **no
block shared**. The script asserts it.

### 4.3 Handling the imbalance: what I compared
All on the same features, scored on **validation lots** (`models/wafer_classifier_metrics.json`):

| Strategy | Macro-F1 | Defects caught | False alarms |
|---|---|---|---|
| unweighted (baseline) | 0.871 | 93.1% | 0.6% |
| balanced weights (1/freq) | 0.861 | 96.3% | 2.0% |
| sqrt-balanced weights | 0.865 | 94.8% | 1.1% |
| undersample "none" + sqrt weights | 0.868 | 96.7% | 2.2% |
| **SMOTE (training fold only)** ✓ | **0.886** | 93.7% | 0.7% |
| random forest, balanced | 0.856 | 91.6% | 0.6% |

**What to say about it:**
- The metric (macro-F1, every class weighted equally) was fixed **before** looking at
  results. Changing it afterwards to favour a result would be cherry-picking.
- Class weighting **trades false alarms for recall**: it catches more defects but
  roughly triples false alarms. Macro-F1 slightly prefers SMOTE here. If the fab says
  a missed defect costs 20× a false alarm, I'd pick `balanced`. That's a business
  decision, and I'd expose it as a setting.
- **SMOTE is legitimate here and wasn't on SECOM.** Here it runs *inside the
  training fold* and is validated on different lots. (See §5.3 for the wrong way.)
- Accuracy is never used to choose anything: 86% of test wafers are `none`, so
  "always none" scores 86%.

### 4.4 Results on held-out test lots (touched once)
| | |
|---|---|
| Macro-F1 (9 classes) | **0.861** |
| Defective wafers caught | **90.8%** |
| False alarms on good wafers | **0.6%** |
| Accuracy (for reference only) | 97.5% |

Per class, F1 ranges from **0.78 (Loc / Donut / Scratch)** to **0.98 (Edge-Ring)**;
Near-full reaches 0.95 despite having only 40 test wafers.

**Where it fails, and why:** most errors are real defects predicted as `none`
(Edge-Loc 176, Loc 118, Scratch 56 of the test wafers). These are faint patterns:
a short scratch or a small cluster barely stands out from background failures.
Loc → Edge-Loc confusion (42) is a location boundary: a cluster near the edge.

### 4.5 Human review: and a bug I caught in my own rule
Low-confidence predictions go to a person. My first rule was "auto-classified wafers
must be ≥95% correct". **It sent 0% of wafers to review**, because the easy `none`
majority makes overall accuracy 97.7% at *any* threshold. Same accuracy trap again.

Fixed rule: measure correctness only on wafers where the model **or** the expert
says "defect", and pick the lowest threshold where those auto-handled calls are ≥95%
correct on validation. The threshold is applied to **calibrated** confidence (§4.7),
which gives **0.81**. On test, **4.6%** of wafers go to review and auto-handled defect
calls are **92.7%** correct, short of the 95% target set on validation. Say so: the
threshold would be re-tuned on live data.

*If asked why it used to be 0.91:* before calibration the raw probabilities were
over-confident, so the same rule landed at a higher number. Calibration changed the
scale, not the policy: the share reviewed barely moved (4.7% → 4.6%).

### 4.6 CNN vs features: tested, not argued
`scripts/train_wafer_cnn.py` trains a small CNN (4 conv blocks, 283k parameters) on the
raw failure maps, resized to 48×48. Resizing to one size also hides the product size
from the CNN. It uses the **same `lot_split()`** and the **same `scores()`** as XGBoost.

| | macro-F1 | defects caught | false alarms |
|---|---|---|---|
| XGBoost, validation | **0.886** | 93.7% | **0.7%** |
| CNN, validation | 0.847 | 94.3% | 1.1% |
| XGBoost, test | **0.861** | 90.8% | **0.6%** |
| CNN, test | 0.840 | **92.3%** | 1.2% |

- **XGBoost wins on the chosen metric**, and the CNN raises about twice the false
  alarms. The CNN is better only on Donut (F1 0.83 vs 0.78).
- **Imbalance for the CNN:** each epoch uses all defects plus a fresh 25% of `none`,
  with sqrt class weights. That made the CNN over-predict defects (2.9% false alarms),
  so its `none` probability is scaled by a factor chosen on validation (×6). Same idea
  as choosing a threshold on validation: 0.832 → 0.847.
- **Caveats, which you should state yourself:** the CNN was trained for 15 epochs on a
  laptop CPU, and XGBoost's test model was refit on train+validation while the CNN's
  wasn't. The validation rows are the apples-to-apples comparison, and XGBoost still
  wins there. A bigger CNN on a GPU may close the gap. Published CNN results on
  WM-811K are often higher, but usually on random splits, which leak lot-mates.
- **Interview line:** "I didn't assume features beat deep learning. I trained a CNN on
  the identical lot-aware split. Under a laptop budget the features won on macro-F1 and
  false alarms, and they're explainable, so I kept them."

### 4.7 Calibration: making "90% confident" mean 90%
The review rule in §4.5 only works if the confidence number is honest. SMOTE and
boosting push probabilities towards 0 and 1, so the raw model sounded surer than it was:

| Test wafers in the 0.8–0.9 confidence bucket | average confidence | actually correct |
|---|---|---|
| before calibration | 86% | **73%** |
| after calibration | 86% | **86%** |

**What I did:** temperature scaling (`core/calibration.py`). One number, **T = 1.45**,
fitted on **validation** by minimising log-loss; T > 1 softens over-confident
probabilities. The live agent applies the same T, so training and serving match.

**Why temperature scaling and not isotonic regression** (which I'd listed as a next step):
- It has **one parameter**, so it can't overfit. Isotonic fits a curve per class, and
  Near-full has only 40 test wafers.
- It **never changes the predicted class**, only the confidence. Macro-F1, recall and the
  confusion matrix are identical before and after, so calibration can't be accused of
  moving the headline numbers.

**Result on test (ECE: average gap between confidence and accuracy, 0 = perfect):**

| | before | after |
|---|---|---|
| All wafers | 0.010 | **0.002** |
| Defect-relevant wafers (model or expert says defect) | 0.099 | **0.065** |

**Say this yourself:** the all-wafer ECE looked fine even before calibration, because the
`none` majority is easy and confidently right. That's the accuracy trap again. The
defect-relevant number is the one that matters. It improved by a third but is still 6.5
points, so on hard wafers the model remains somewhat over-confident.

### 4.8 Drift monitoring: knowing when the test score stops applying
A new product, tester or process change shifts the inputs, and the test metrics quietly
stop applying. Nobody gets an error. `core/drift.py` watches for that:

- At training time, each of the 26 features is cut into 10 buckets at the training
  deciles and the share per bucket is saved (`models/drift_reference.json`).
- Every live wafer's features are logged (SQLite). The most recent 500 (minimum 50) are
  compared with the reference using the **Population Stability Index**:
  PSI < 0.10 stable, 0.10–0.25 moderate shift, > 0.25 major shift.
- It's exposed at `GET /api/drift` and in the dashboard's Model card tab.

**Why PSI and not a KS test:** with hundreds of wafers, KS calls tiny harmless differences
"significant". PSI measures *how big* the shift is, which is the question that matters.
**Why inputs, not accuracy:** it needs no labels, so drift shows up the day it starts,
weeks before anyone could measure accuracy on new wafers.

**Two sanity checks run at training time** (prove the monitor isn't broken in either direction):

| Check | Expected | Result |
|---|---|---|
| Held-out test lots vs training | should be stable (same fab, same period) | **stable**, max PSI 0.064 |
| Only defective test wafers vs training | should alarm (a very different mix) | **major**, max PSI 3.4 on cluster features |

*If asked "isn't PSI 3.4 a problem?":* no. That check deliberately feeds the monitor a
skewed sample: training is 85% clean wafers, this sample is 100% defective, so cluster
features change completely. It proves the alarm fires. A monitor that never fires
proves nothing.

**Limit to state:** drift says the inputs changed, not that accuracy dropped. A major
shift should trigger relabelling a sample and re-evaluating, not an automatic retrain.

### 4.9 In progress: self-supervised pre-training on the unlabelled wafers
WM-811K has ~638,000 wafers nobody labelled. Labels are the real bottleneck in a fab, so
the next experiment tests whether those wafers can make the CNN competitive:

- `scripts/download_unlabelled.py` keeps each wafer's 48×48 failure map (empty maps dropped).
- `scripts/pretrain_wafer_cnn.py` runs **SimCLR**: each wafer is shown twice with changes
  that don't alter the defect (rotate / mirror, drop 15% of failing cells, add a few
  random ones), and the CNN learns to place the two versions close together. **No
  shifting or cropping**, because position is the label (Edge-Loc vs Loc).
- **No leakage:** unlabelled wafers from validation or test lot-blocks are dropped, so
  test lots stay unseen in every form. Labels are used only to *monitor* pre-training
  (k-NN macro-F1 on validation), never to train or choose.
- `scripts/train_wafer_cnn.py --init <encoder> --label-fraction 0.1 --tag …` fine-tunes from
  the pre-trained encoder, optionally with only 10% of the labelled lots. That measures
  the question a fab cares about: *how much does pre-training help when labels are scarce?*
- It needs a GPU, so it runs on Colab (`notebooks/wafer_cnn_colab.ipynb`, which clones the
  repo from GitHub).

**Status: no results yet.** Don't quote numbers until the run is done. *Interview line:*
"The feature model wins today; the experiment I set up next is whether self-supervised
pre-training on the 638k unlabelled wafers changes that, especially with few labels."

---

## 5. Process monitor (SECOM): the honest negative result

### 5.1 Setup
- **Chronological split:** first 70% of runs (Jul 19 → Sep 26) train, last 30%
  (Sep 26 → Oct 17) test. A random split lets the model learn from the future.
- 590 sensors → **444 kept**: 116 never change, 28 are more than half empty. That's
  decided on training runs only, inside a scikit-learn pipeline (`core/secom.py`).
- Six imbalance strategies compared with **time-series cross-validation inside
  the training months**: logistic regression (plain / class-weighted / SMOTE),
  random forest (balanced), XGBoost (`scale_pos_weight`), Isolation Forest
  (unsupervised, fit on passing runs only). Metric: **PR-AUC**
- **Selected:** balanced random forest (CV PR-AUC 0.162 vs 0.071 random baseline).

### 5.2 Test result
| | |
|---|---|
| PR-AUC | **0.069** (95% CI 0.043–0.12) vs **0.055** random |
| ROC-AUC | **0.58** (95% CI 0.47–0.68) |
| At a 10% false-alarm budget | catches **2 of 26** failures |

The confidence intervals include "random". **On future months, this model barely works.**
The failure rate drifts from 22% (July) to 3% (September): the process changed,
and a model trained on the past doesn't transfer.

### 5.3 Why tutorials report ~99% on SECOM (same model family, same data)
| Evaluation | ROC-AUC |
|---|---|
| SMOTE on the whole dataset, **then** random split (the common tutorial) | **1.000** |
| Random split, no leakage | 0.80 (0.70 with another seed: unstable) |
| **Chronological split (what I report)** | **0.58** |

SMOTE creates synthetic failures by blending real ones. Done *before* the split,
near-copies of test failures end up in training, so the model is graded on
answers it has seen. **This comparison is one of the strongest things you can show
in an interview**: it proves you know how evaluation goes wrong.

### 5.4 What the product does with a weak model
- Its score is shown as **"process risk (low-confidence)"** and **never escalates
  anything on its own**.
- The useful part of SECOM is **SPC** (statistical process control): which sensors are
  outside mean ± 3σ of good training runs. It needs no labels, it's how fabs actually
  monitor tools, and it gives the root-cause agent concrete evidence.
- **Multiple-comparisons catch:** with 444 sensors, ~1 is outside 3σ on a perfectly
  normal run by chance (0.27% × 444). So a run is "unusual" only when more sensors
  are out than on 95% of good runs.

---

## 6. The agents: what each one decides and why

| # | Agent | Decides with | Why that tool |
|---|---|---|---|
| 1 | Data Ingestor | validation rules | Rejects malformed wafer maps (ragged, values ≠ 0/1/2, no die) — garbage maps would silently give garbage features |
| 2 | Sensor Parser | SPC (mean ± 3σ) | Standard, explainable, label-free |
| 3 | Defect Classifier | XGBoost + calibrated confidence | 173k labelled examples; measurable; fast. Logs each wafer's features for the drift monitor |
| 4 | Root Cause | ChromaDB similar-wafer search + LLM | The LLM *explains*, it doesn't *decide*. It must choose among documented causes for the pattern (`core/defect_knowledge.py`) and must not invent tools or events |
| 5 | Trend Detector | rules over the system's own history + SECOM score | Repeats across lot / tool / material are what make a defect *systemic* |
| 6 | Supplier Alert | evidence gate → LLM writes the alert | Suppliers are only contacted with evidence (below) |
| 7 | Compliance | LLM | Turns the evidence into an ISO 9001 CAPA document |

### Root cause retrieval: why vectors of *features*, not text
- The old version embedded text descriptions with a language model. That was slow
  (the embedding model took ~4 minutes to load on first use) and compared *words*,
  not wafers.
- Now ChromaDB stores **40k historical wafers as their 26 standardised features** and
  returns the 10 most similar. Their expert labels are an **independent second
  opinion**: "9/10 similar wafers were also Edge-Ring" vs "only 3/10 agree": the
  LLM is told to lower its confidence when neighbours disagree.
- **Only training lots are indexed.** Demo wafers come from test lots, so a wafer can
  never "find itself" in history.
- Pattern → likely-cause facts are a **lookup table**, not vector search: there are 9
  patterns and the classifier already says which one, so retrieval would only add
  latency and a way to fetch the wrong document.

### Trend rules: why each threshold
| Pattern | Rule | Reasoning |
|---|---|---|
| Lot | ≥3 wafers in the lot share the pattern | One or two can be noise |
| Tool | same *systematic* pattern ≥3 times on one tool **across ≥2 lots** | One bad lot on a tool is a lot problem; two lots on one tool points at the tool |
| Material | same material lot defective on **≥2 tools** and at ≥2× plant-wide defect rate | Several tools, one common material → the material is the common factor |

These run over **the system's own inspection history** (`core/history.py`, SQLite),
because the datasets have no tool or supplier IDs. Inventing a fake history to make
the demo look busy would be fabricating data; instead, the batch scenarios build
history live from real held-out lots.

### Supplier alert: the evidence gate
A supplier is contacted only if (a) the same material lot fails on several tools, or
(b) the pattern *can* be caused by incoming material (Random, Near-full) **and**
root-cause analysis concluded it was. An Edge-Ring on one wafer is a tool problem,
and the old version would have emailed the supplier for it. A test checks this.

---

## 7. Questions you'll likely get

**"Why not just use accuracy?"** Because 86% of wafers are `none`; a model that never
finds a defect scores 86%. I use macro-F1 and defect recall.

**"How do you know your test score isn't inflated?"** Split by lot (measured 60% vs 24%
neighbour similarity proves lots cluster), the test set was evaluated once, strategy
and threshold were chosen on a separate validation set, and the similar-wafer index
excludes test lots.

**"SMOTE causes leakage, doesn't it?"** Only if applied before splitting. I show both:
on SECOM, SMOTE-before-split gives a fake 1.000; on WM-811K, SMOTE inside the training
fold is validated on unseen lots and is genuinely the best strategy.

**"Your SECOM model doesn't work. Why keep it?"** It's reported honestly as low-confidence
and never drives a decision. The SPC part of that pipeline *is* useful. Showing a
negative result with confidence intervals is more credible than a suspicious 99%.

**"Why an LLM at all?"** Only to explain and to write documents. Every *decision*
(pattern, escalation, whether to contact a supplier) is made by models or explicit
rules that can be tested. That's on purpose: LLM decisions are hard to test and audit.

**"How do you know the confidence numbers mean anything?"** They're calibrated with
temperature scaling fitted on validation. On test, wafers the model calls 86% certain are
right 86% of the time (73% before). On defect-relevant wafers a 6.5-point gap remains (§4.7).

**"How would you know if the model breaks in production?"** The drift monitor compares
live wafers' features with the training data (PSI), with no labels needed. I checked it
both ways: stable on held-out lots, alarms on a deliberately skewed sample (§4.8).

**"What would you do next?"**
1. Finish the self-supervised pre-training run on the ~638,000 *unlabelled* wafers
   (already built, §4.9) and report whether it helps, especially with 10% of the labels.
2. Act on drift: when PSI goes major, relabel a sample of recent wafers and re-evaluate
   before retraining.
3. Retrain SECOM on a rolling window: the July→October shift is its real problem.
4. Cost-sensitive selection once the fab provides the cost of a miss vs a false alarm.
5. Postgres instead of SQLite for history and the drift log when running more than one worker.

**"What are the limitations?"**
- Datasets from different fabs; tool / supplier IDs simulated.
- WM-811K labels are expert judgments, so some are ambiguous (e.g. small Loc vs none).
- Die positions on rebuilt maps are approximated by an inscribed ellipse (the raw data
  gives failing die exactly, not every passing die), applied identically in training and serving.
- SECOM sensors are anonymised, so SPC can say *which* sensor but not *what it measures*.

---

## 8. Where everything lives

| What | File |
|---|---|
| Download + verify data | `scripts/download_data.py` |
| Wafer features | `core/wafer_features.py`, `scripts/build_wafer_features.py` |
| Classifier training + strategy comparison | `scripts/train_wafer_classifier.py` |
| SECOM training + leakage contrast | `scripts/train_process_monitor.py`, `core/secom.py` |
| Pattern knowledge | `core/defect_knowledge.py` |
| CNN comparison | `scripts/train_wafer_cnn.py` |
| Calibration | `core/calibration.py` (T fitted in `scripts/train_wafer_classifier.py`) |
| Drift monitor | `core/drift.py`, `models/drift_reference.json`, `GET /api/drift` |
| Self-supervised pre-training (GPU) | `notebooks/wafer_cnn_colab.ipynb`, `scripts/download_unlabelled.py`, `scripts/pretrain_wafer_cnn.py` |
| Data-preparation walk-through | `docs/DATA_PREPARATION.md` |
| Inspection history | `core/history.py` |
| All metrics (shown in the dashboard's Model card tab) | `models/*_metrics.json` |
| Tests | `tests/test_pipeline.py` |
| Old AI4I version | `scripts/legacy_ai4i/`, `models/legacy_ai4i/` |
