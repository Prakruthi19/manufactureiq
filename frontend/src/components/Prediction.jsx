// Predicted vs actual for one held-out wafer, plus the model's full probability
// spread and the process (SECOM) readout.

const pct = (v) => `${Math.round((v || 0) * 100)}%`

export default function Prediction({ state, truth, secom }) {
  const clf = state.classification
  const spc = state.sensor_summary?.spc
  const wafer = state.sensor_summary?.wafer
  const trend = state.trend
  const predicted = clf.pattern
  const correct = truth && predicted === truth
  const probs = Object.entries(clf.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 4)

  return (
    <div className="card prediction">
      <div className="card-head">
        <h2>Prediction vs expert label</h2>
        {truth && <span className={`chip ${correct ? 'chip-ok' : 'chip-warn'}`}>{correct ? '✓ correct' : '✗ mismatch'}</span>}
      </div>
      <div className="pred-grid">
        <div>
          <div className="pred-label">Model says</div>
          <div className="pred-value">{predicted}</div>
          <div className="pred-sub">{pct(clf.confidence_score)} confidence · review below {pct(clf.review_threshold)}</div>
        </div>
        <div>
          <div className="pred-label">Expert label (held out)</div>
          <div className="pred-value">{truth ?? '—'}</div>
          <div className="pred-sub">WM-811K ground truth</div>
        </div>
        <div className="probs">
          {probs.map(([c, p]) => (
            <div key={c} className="prob-row">
              <span className="prob-name">{c}</span>
              <span className="prob-track"><span className="prob-fill" style={{ width: pct(p) }} /></span>
              <span className="prob-val">{pct(p)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="tiles">
        <Tile label="Yield" value={wafer ? `${wafer.yield_pct}%` : '—'} />
        <Tile label="Failing die" value={wafer ? `${wafer.failing_die}/${wafer.die}` : '—'} />
        <Tile label="SPC out of control" value={spc ? `${spc.out_of_control} (normal ≤${Math.round(spc.typical_out_of_control_p95)})` : '—'} />
        <Tile label="Process risk*" value={trend?.process_risk != null ? trend.process_risk.toFixed(2) : '—'} />
      </div>
      <p className="batch-hint">
        *Process risk comes from the SECOM model, which is only slightly better than chance on future
        runs (see Model card), so it never escalates on its own.
        {secom && ` This run actually ${secom.failed_line_test ? 'FAILED' : 'passed'} its line test.`}
      </p>
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
