import { useEffect, useState } from 'react'
import { getModelCard } from '../api'
import DriftPanel from './DriftPanel'
import ReliabilityChart from './ReliabilityChart'

// Shows models/*_metrics.json exactly as the training scripts wrote them —
// the dashboard never states a number the evaluation didn't produce.

const f3 = (v) => (v == null ? '—' : Number(v).toFixed(3))
const pc = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

export default function ModelCard() {
  const [card, setCard] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => { getModelCard().then(setCard).catch((e) => setError(e.message)) }, [])

  if (error) return <div className="card"><div className="alert error">{error}</div></div>
  if (!card) return <div className="card"><p className="batch-note">Loading…</p></div>
  const wc = card.wafer_classifier
  const pm = card.process_monitor
  const noneShare = wc && wc.test_per_class.none.support / wc.split.test

  return (
    <div className="model-card">
      {wc && (
        <section className="card">
          <div className="card-head"><h2>Defect Classifier — WM-811K</h2><span className="chip">{wc.selected_strategy}</span></div>
          <p className="prose">{wc.dataset}. Split by lot-block so no lot appears in both train and test
            ({wc.split.train.toLocaleString()} / {wc.split.validation.toLocaleString()} / {wc.split.test.toLocaleString()} wafers).
            Strategy chosen by macro-F1 on validation; test touched once.</p>

          <div className="tiles">
            <Tile label="Macro-F1 (9 classes)" value={f3(wc.test.macro_f1)} />
            <Tile label="Defective wafers caught" value={pc(wc.test.defect_recall)} />
            <Tile label="False alarms on good wafers" value={pc(wc.test.false_alarm_rate)} />
            <Tile label="Sent to human review" value={pc(wc.test.human_review_share)} />
          </div>
          <p className="batch-hint">Accuracy is {pc(wc.test.accuracy)} — but {pc(noneShare)} of test wafers are "none", so a model that never finds a defect would score {pc(noneShare)}. That's why it isn't used for decisions.
            Auto-classified wafers (confidence ≥ {pc(wc.review_threshold)}) are {pc(wc.test.auto_classified_accuracy)} correct.</p>

          <h3>Imbalance strategies (validation lots)</h3>
          <Table head={['Strategy', 'Macro-F1', 'Patterns F1', 'Defect recall', 'False alarms', 'Accuracy']}
            rows={Object.entries(wc.strategy_comparison_validation).map(([k, s]) => [
              k === wc.selected_strategy ? `▶ ${k}` : k, f3(s.macro_f1), f3(s.macro_f1_patterns),
              pc(s.defect_recall), pc(s.false_alarm_rate), pc(s.accuracy)])} />

          <h3>Per class (held-out test lots)</h3>
          <Table head={['Class', 'Precision', 'Recall', 'F1', 'Test wafers']}
            rows={Object.entries(wc.test_per_class).map(([c, m]) => [c, f3(m.precision), f3(m.recall), f3(m.f1), m.support.toLocaleString()])} />

          {wc.calibration && <Calibration c={wc.calibration} threshold={wc.review_threshold} />}
          {wc.drift_checks && <DriftPanel checks={wc.drift_checks} />}

          <h3>Confusion matrix (rows = expert, columns = model)</h3>
          <div className="table-wrap">
            <table className="results-table cm">
              <thead><tr><th></th>{wc.classes.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{wc.test_confusion_matrix.map((row, i) => (
                <tr key={i}><th>{wc.classes[i]}</th>{row.map((v, j) => (
                  <td key={j} className={i === j ? 'cm-diag' : v ? 'cm-off' : ''}>{v || ''}</td>))}</tr>))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {pm && (
        <section className="card">
          <div className="card-head"><h2>Process monitor — SECOM</h2><span className="chip chip-warn">weak signal</span></div>
          <p className="prose">{pm.dataset}. Chronological split: train {pm.split.train[0].slice(0, 10)} → {pm.split.train[1].slice(0, 10)},
            test {pm.split.test[0].slice(0, 10)} → {pm.split.test[1].slice(0, 10)} ({pm.split.test_failures} failures).
            {' '}{pm.sensors_kept} of 590 sensors kept.</p>
          <div className="tiles">
            <Tile label="Test PR-AUC" value={`${f3(pm.test.pr_auc)}`} />
            <Tile label="Random baseline" value={f3(pm.test.random_baseline_pr_auc)} />
            <Tile label="Test ROC-AUC" value={f3(pm.test.roc_auc)} />
            <Tile label="Failures caught @10% alarms" value={pm.test.at_threshold.failures_caught} />
          </div>
          <p className="batch-hint">95% CI: PR-AUC {pm.test.pr_auc_ci95.join('–')}, ROC-AUC {pm.test.roc_auc_ci95.join('–')}. {pm.decision}</p>

          <h3>Imbalance strategies (time-series CV, training months)</h3>
          <Table head={['Strategy', 'PR-AUC', '± std', 'ROC-AUC']}
            rows={Object.entries(pm.cv_comparison).map(([k, s]) => [k === pm.selected ? `▶ ${k}` : k, f3(s.pr_auc), f3(s.pr_auc_std), f3(s.roc_auc)])} />

          <h3>Same data, different evaluation</h3>
          <Table head={['How it was evaluated', 'ROC-AUC']}
            rows={[
              ['SMOTE on all data, then random split (common tutorial)', f3(pm.evaluation_contrast_roc_auc.smote_before_random_split)],
              ['Random split, no leakage', f3(pm.evaluation_contrast_roc_auc.random_split_no_leak)],
              ['Chronological split (what we report)', f3(pm.evaluation_contrast_roc_auc.chronological_split)],
            ]} />
          <p className="batch-hint">SMOTE builds synthetic failures by blending real ones; doing it before the split puts
            near-copies of test rows into training. The ~1.0 is leakage, not skill.</p>
        </section>
      )}
    </div>
  )
}

function Calibration({ c, threshold }) {
  return (
    <>
      <h3>Calibration: does "90% confident" mean right 90% of the time?</h3>
      <p className="batch-hint">SMOTE and boosting make raw probabilities over-confident. One number, the temperature
        T = {c.temperature.toFixed(2)}, was fitted on validation lots to soften them. It never changes which pattern is
        predicted, only how sure the model says it is. The human-review cut-off ({pc(threshold)}) is set on these calibrated values.</p>
      <div className="calib-grid">
        <ReliabilityChart calibration={c} />
        <div>
          <div className="tiles">
            <Tile label="Calibration error, all test wafers (before)" value={pc(c.test_ece_before)} />
            <Tile label="Calibration error, all test wafers (after)" value={pc(c.test_ece_after)} />
            <Tile label="Defect-relevant wafers (before)" value={pc(c.test_ece_defect_relevant_before)} />
            <Tile label="Defect-relevant wafers (after)" value={pc(c.test_ece_defect_relevant_after)} />
          </div>
          <p className="batch-hint">Calibration error (ECE) is the average gap between stated confidence and actual accuracy.
            The all-wafer figure is flattered by the easy "none" wafers; the defect-relevant row (wafers that are, or are
            predicted to be, defective) is the honest one.</p>
        </div>
      </div>
      <Table head={['Confidence bucket', 'Wafers', 'Stated (after)', 'Actually correct']}
        rows={c.bins_after.filter((b) => b.n).map((b) => [`${pc(b.lo)}–${pc(b.hi)}`, b.n.toLocaleString(), pc(b.confidence), pc(b.accuracy)])} />
    </>
  )
}

function Table({ head, rows }) {
  return (
    <div className="table-wrap">
      <table className="results-table">
        <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className={j ? 'mono' : ''}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}

function Tile({ label, value }) {
  return (
    <div className="tile">
      <div className="tile-value">{value}</div>
      <div className="tile-label">{label}</div>
    </div>
  )
}
