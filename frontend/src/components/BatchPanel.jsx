import { useEffect, useRef, useState } from 'react'
import { submitInspection, getInspection, clearHistory } from '../api'
import ProgressBar from './ProgressBar'
import WaferMap from './WaferMap'

// Lot scenarios: consecutive wafers from REAL held-out WM-811K lots, run through
// the ML-only pipeline (no LLM calls) in order, so the Trend Detector's lot /
// tool / material history builds up wafer by wafer.

const PATTERN_LABEL = { batch_pattern: 'lot', machine_pattern: 'tool', supplier_pattern: 'material', isolated: '—' }

export default function BatchPanel({ scenarios, note }) {
  const [items, setItems] = useState([])
  const [active, setActive] = useState(null)
  const [running, setRunning] = useState(false)
  const [msg, setMsg] = useState(null)
  const timerRef = useRef(null)

  async function run(idx) {
    const sc = scenarios[idx]
    setActive(idx)
    setRunning(true)
    setMsg(null)
    const submitted = []
    // Sequential submission keeps queue order = lot order (history depends on it).
    for (const it of sc.items) {
      try {
        const { task_id } = await submitInspection(it.inspection, [], false)
        submitted.push({ ...it, taskId: task_id, state: 'PENDING', result: null })
      } catch {
        submitted.push({ ...it, taskId: null, state: 'FAILURE', result: null })
      }
    }
    setItems(submitted)
  }

  async function reset() {
    await clearHistory()
    setItems([])
    setActive(null)
    setMsg('Inspection history cleared — scenarios start from a clean slate.')
  }

  useEffect(() => {
    if (!items.length) return
    const pending = items.some((it) => it.taskId && !['SUCCESS', 'FAILURE'].includes(it.state))
    if (!pending) { setRunning(false); return }
    timerRef.current = setInterval(async () => {
      const updated = await Promise.all(items.map(async (it) => {
        if (!it.taskId || ['SUCCESS', 'FAILURE'].includes(it.state)) return it
        try {
          const d = await getInspection(it.taskId)
          return { ...it, state: d.state, result: d.result || it.result }
        } catch { return it }
      }))
      setItems(updated)
    }, 800)
    return () => clearInterval(timerRef.current)
  }, [items])

  if (!scenarios) return <div className="card"><p className="batch-note">Loading scenarios…</p></div>

  const settled = items.filter((it) => ['SUCCESS', 'FAILURE'].includes(it.state))
  const scored = settled.filter((it) => it.result?.classification)
  const correct = scored.filter((it) => it.result.classification.pattern === it.true_pattern).length

  return (
    <div className="card">
      <div className="card-head">
        <h2>Lot scenarios</h2>
        <div className="head-actions">
          {items.length > 0 && <span className="chip chip-ok">{settled.length}/{items.length} done</span>}
          {scored.length > 0 && <span className="chip">{correct}/{scored.length} match expert label</span>}
          <button type="button" className="link" onClick={reset} disabled={running}>Reset history</button>
        </div>
      </div>
      <p className="batch-hint">{note} Batch runs skip the LLM agents (fast, free); run a single wafer for the full report.</p>
      {msg && <p className="batch-note">{msg}</p>}

      <div className="scenarios">
        {scenarios.map((sc, i) => (
          <div key={i} className={`scenario ${active === i ? 'active' : ''}`}>
            <h3>{sc.name}</h3>
            <p>{sc.story}</p>
            <p className="batch-hint">
              {sc.items.length} wafers · {sc.lot} · true labels: {Object.entries(sc.counts).map(([k, v]) => `${k} ${v}`).join(', ')}
            </p>
            <button className="submit inline" onClick={() => run(i)} disabled={running}>
              {running && active === i ? 'Running…' : 'Run this lot'}
            </button>
          </div>
        ))}
      </div>

      {items.length > 0 && (<>
        <ProgressBar value={settled.length / items.length} label="Lot progress" />
        <div className="table-wrap">
          <table className="results-table">
            <thead>
              <tr><th>#</th><th>Wafer</th><th>Map</th><th>Tool</th><th>Expert</th><th>Model</th><th>Conf.</th><th>Trend</th><th>Escalate</th></tr>
            </thead>
            <tbody>
              {items.map((it, i) => {
                const clf = it.result?.classification
                const tr = it.result?.trend
                const ok = clf && clf.pattern === it.true_pattern
                return (
                  <tr key={i}>
                    <td className="mono">{i + 1}</td>
                    <td className="mono">{it.inspection.wafer_id}</td>
                    <td><WaferMap grid={it.inspection.wafer_map} size={40} /></td>
                    <td className="mono">{it.inspection.machine_id}</td>
                    <td>{it.true_pattern}</td>
                    <td className={clf ? (ok ? 'cell-ok' : 'cell-bad') : ''}>
                      {clf ? clf.pattern : <span className="node-spinner sm" />}
                    </td>
                    <td className="mono">{clf ? `${Math.round(clf.confidence_score * 100)}%` : ''}</td>
                    <td title={tr?.summary}>{tr ? PATTERN_LABEL[tr.pattern] ?? tr.pattern : ''}</td>
                    <td>{tr ? (tr.escalate ? <span className="badge sev-high">yes</span> : 'no') : ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {(() => {
          const last = [...items].reverse().find((it) => it.result?.trend?.escalate)
          return last && <div className="alert warn"><strong>Trend detector:</strong> {last.result.trend.summary}</div>
        })()}
      </>)}
    </div>
  )
}
