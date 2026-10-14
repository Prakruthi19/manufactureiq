import { useEffect, useState } from 'react'
import { getDrift } from '../api'

// Live input drift: recent wafers' features vs the training data (PSI).
// Status always ships as icon + label, never colour alone.

const STATUS = {
  stable: { icon: '✓', label: 'Stable', cls: 'ok' },
  moderate: { icon: '!', label: 'Moderate shift', cls: 'warn' },
  major: { icon: '✕', label: 'Major shift', cls: 'fail' },
}

export default function DriftPanel({ checks }) {
  const [live, setLive] = useState(null)
  const [error, setError] = useState(null)
  const load = () => getDrift().then(setLive).catch((e) => setError(e.message))
  useEffect(() => { load() }, [])

  return (
    <>
      <h3>Input drift: are today's wafers like the training wafers?</h3>
      <p className="batch-hint">
        Each of the 26 features is compared with the training data using the Population Stability
        Index (PSI). Below 0.10 is stable, 0.10–0.25 a moderate shift, above 0.25 a major shift.
        It needs no labels, so a new product or process change shows up the day it starts.
      </p>

      {checks && (
        <div className="table-wrap">
          <table className="results-table">
            <thead><tr><th>Sanity check (offline)</th><th>Status</th><th>Max PSI</th><th>Most shifted feature</th></tr></thead>
            <tbody>
              {[['Test lots vs training', checks.test_lots_vs_training],
                ['Only defective wafers vs training', checks.defective_wafers_only_vs_training]].map(([name, c]) => (
                <tr key={name}><td>{name}</td><td><Status s={c.status} /></td>
                  <td className="mono">{c.max_psi.toFixed(3)}</td><td className="mono">{c.features[0].feature}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {checks && <p className="batch-hint">Unseen lots from the same fab read as stable; a stream of only defective
        wafers is flagged. That shows the monitor tells a real shift from normal variation.</p>}

      <div className="drift-live">
        <strong>Live: </strong>
        {error && <span className="alert error">{error}</span>}
        {!live && !error && <span>Loading…</span>}
        {live && live.status === 'not_enough_data' && (
          <span>{live.n_wafers} of {live.min_wafers} wafers inspected so far. Run a lot scenario to collect more.</span>
        )}
        {live && live.status === 'unavailable' && <span>{live.detail}</span>}
        {live && STATUS[live.status] && (
          <span><Status s={live.status} /> over the last {live.n_wafers} wafers. Max PSI {live.max_psi.toFixed(3)} ({live.features[0].feature}).
            {live.status !== 'stable' && ' The demo lot scenarios are hand-picked defective wafers, so a shift is expected here.'}</span>
        )}
        <button type="button" className="link" onClick={load}>Refresh</button>
      </div>
      {live?.features && (
        <div className="table-wrap">
          <table className="results-table">
            <thead><tr><th>Feature</th><th>PSI</th><th>Status</th></tr></thead>
            <tbody>{live.features.slice(0, 5).map((f) => (
              <tr key={f.feature}><td className="mono">{f.feature}</td><td className="mono">{f.psi.toFixed(3)}</td>
                <td><Status s={f.psi < 0.1 ? 'stable' : f.psi < 0.25 ? 'moderate' : 'major'} /></td></tr>))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

function Status({ s }) {
  const st = STATUS[s]
  return <span className={`runpill ${st.cls}`}><span aria-hidden="true">{st.icon}</span> {st.label}</span>
}
