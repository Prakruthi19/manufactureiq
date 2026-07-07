import { useEffect, useRef, useState } from 'react'
import { submitInspection, getInspection } from '../api'
import { parseCsv, rowToPayload, BATCH_LIMIT } from '../batch'
import ProgressBar from './ProgressBar'
import AgentDetails from './AgentDetails'

const SEVERITY_CLASS = { low: 'sev-low', medium: 'sev-medium', high: 'sev-high', critical: 'sev-critical' }

// Escape a value for CSV (quote if it contains comma/quote/newline).
const csvCell = (v) => {
  const s = String(v ?? '')
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s
}

export default function BatchPanel() {
  const [items, setItems] = useState([]) // [{label, product, taskId, state, result}]
  const [fileName, setFileName] = useState(null)
  const [pending, setPending] = useState([]) // parsed payloads not yet run
  const [running, setRunning] = useState(false)
  const [note, setNote] = useState(null)
  const [openIdx, setOpenIdx] = useState(null) // which row's full report is expanded
  const timerRef = useRef(null)

  async function onFile(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    setItems([])
    const text = await file.text()
    const rows = parseCsv(text)
    let payloads = rows.map(rowToPayload)
    if (payloads.length > BATCH_LIMIT) {
      setNote(`File has ${payloads.length} rows — running the first ${BATCH_LIMIT}.`)
      payloads = payloads.slice(0, BATCH_LIMIT)
    } else {
      setNote(`${payloads.length} record(s) loaded.`)
    }
    setPending(payloads)
  }

  async function runBatch() {
    if (!pending.length) return
    setRunning(true)
    const submitted = []
    for (const p of pending) {
      try {
        // Batch uses the fast dry pipeline (full=false) — bulk classification only.
        const { task_id } = await submitInspection(p.inspection, p.sensors, true)
        submitted.push({ label: p.label, product: p.product, taskId: task_id, state: 'PENDING', result: null })
      } catch {
        submitted.push({ label: p.label, product: p.product, taskId: null, state: 'FAILURE', result: null })
      }
    }
    setItems(submitted)
  }

  // Poll all in-flight tasks until every one settles.
  useEffect(() => {
    if (!items.length) return
    const anyPending = items.some((it) => it.taskId && it.state !== 'SUCCESS' && it.state !== 'FAILURE')
    if (!anyPending) {
      setRunning(false)
      clearInterval(timerRef.current)
      return
    }
    timerRef.current = setInterval(async () => {
      const updated = await Promise.all(
        items.map(async (it) => {
          if (!it.taskId || it.state === 'SUCCESS' || it.state === 'FAILURE') return it
          try {
            const data = await getInspection(it.taskId)
            return { ...it, state: data.state, result: data.result || it.result }
          } catch {
            return it
          }
        })
      )
      setItems(updated)
    }, 800)
    return () => clearInterval(timerRef.current)
  }, [items])

  const done = items.filter((it) => it.state === 'SUCCESS' || it.state === 'FAILURE').length
  const progress = items.length ? done / items.length : 0

  function downloadCsv() {
    const header = ['record', 'product', 'defect_type', 'severity', 'confidence', 'status']
    const rows = items.map((it) => {
      const clf = it.result?.classification
      return [
        it.label,
        it.product,
        clf?.defect_type ?? '',
        clf?.severity ?? '',
        clf ? `${Math.round((clf.confidence_score || 0) * 100)}%` : '',
        it.state,
      ]
    })
    const csv = [header, ...rows].map((r) => r.map(csvCell).join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    const a = document.createElement('a')
    a.href = url
    a.download = 'batch_results.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="card">
      <div className="card-head">
        <h2>Batch inspection</h2>
        <div className="head-actions">
          {items.length > 0 && <span className="chip chip-ok">{done}/{items.length} done</span>}
          {items.length > 0 && <button type="button" className="link" onClick={downloadCsv}>Download CSV</button>}
        </div>
      </div>

      <div className="batch-controls">
        <label className="file-btn">
          {fileName || 'Choose CSV file…'}
          <input type="file" accept=".csv,text/csv" onChange={onFile} hidden />
        </label>
        <button className="submit inline" onClick={runBatch} disabled={!pending.length || running}>
          {running ? 'Running…' : `Run ${pending.length || ''} record(s)`}
        </button>
      </div>
      {note && <p className="batch-note">{note}</p>}
      <p className="batch-hint">Expected columns (AI4I 2020): Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].</p>

      {items.length > 0 && (
        <>
          <ProgressBar value={progress} label="Batch progress" />
          <div className="table-wrap">
            <table className="results-table">
              <thead>
                <tr><th>Record</th><th>Product</th><th>Defect</th><th>Severity</th><th>Conf.</th><th>Status</th><th>Report</th></tr>
              </thead>
              <tbody>
                {items.map((it, i) => {
                  const clf = it.result?.classification
                  const settled = it.state === 'SUCCESS' || it.state === 'FAILURE'
                  const hasReport = it.state === 'SUCCESS' && it.result
                  return (
                    <tr key={i}>
                      <td className="mono">{it.label}</td>
                      <td className="mono">{it.product}</td>
                      <td>{clf ? clf.defect_type?.replaceAll('_', ' ') : '—'}</td>
                      <td>{clf ? <span className={`badge ${SEVERITY_CLASS[clf.severity] || ''}`}>{clf.severity}</span> : '—'}</td>
                      <td className="mono">{clf ? `${Math.round((clf.confidence_score || 0) * 100)}%` : '—'}</td>
                      <td>{settled
                        ? <span className={`dotstate ${it.state === 'SUCCESS' ? 'ok' : 'fail'}`}>{it.state === 'SUCCESS' ? 'done' : 'failed'}</span>
                        : <span className="node-spinner sm" />}</td>
                      <td>{hasReport
                        ? <button type="button" className="link" onClick={() => setOpenIdx(openIdx === i ? null : i)}>{openIdx === i ? 'Hide' : 'View'}</button>
                        : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {openIdx != null && items[openIdx]?.result && (
            <div className="batch-detail">
              <div className="card-head">
                <h3>Report — {items[openIdx].label} · {items[openIdx].product}</h3>
                <button type="button" className="link" onClick={() => setOpenIdx(null)}>Close</button>
              </div>
              <AgentDetails state={items[openIdx].result} />
            </div>
          )}
        </>
      )}
    </div>
  )
}
