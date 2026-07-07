import { useEffect, useRef, useState } from 'react'
import { submitInspection, getInspection } from './api'
import { AGENTS, TIERS, finalStatusFor, sleep } from './agents'
import AgentPipeline from './components/AgentPipeline'
import Telemetry from './components/Telemetry'
import AgentDetails from './components/AgentDetails'
import BatchPanel from './components/BatchPanel'

const SAMPLE_INSPECTION = {
  record_id: 'REC-001',
  batch_id: 'BATCH-0146',
  product_id: 'M014860',
  machine_id: 'MCH-002',
  timestamp: '2026-07-06T09:00:00',
  raw_notes: 'Power failure detected. Torque spike at 1741 rpm.',
}
const SAMPLE_SENSOR = {
  sensor_id: 'MCH-002-S1',
  temperature_c: 27.6,
  machine_speed_rpm: 1741,
  anomaly_flagged: true,
}

// All 7 agents run in the full pipeline — every one starts queued.
const initStatus = () => Object.fromEntries(AGENTS.map((a) => [a.key, 'queued']))

export default function App() {
  const [mode, setMode] = useState('single') // 'single' | 'batch'
  const [insp, setInsp] = useState(SAMPLE_INSPECTION)
  const [sensor, setSensor] = useState(SAMPLE_SENSOR)
  const [taskId, setTaskId] = useState(null)
  const [poll, setPoll] = useState(null)
  const [error, setError] = useState(null)
  const [agentStatus, setAgentStatus] = useState(initStatus())
  const [revealing, setRevealing] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  const timerRef = useRef(null)
  const clockRef = useRef(null)
  const revealedFor = useRef(null) // guard: only reveal once per task

  // Elapsed clock while a run is active
  useEffect(() => {
    const active = taskId && !poll?.result && poll?.state !== 'FAILURE'
    if (active) {
      const start = Date.now()
      clockRef.current = setInterval(() => setElapsed((Date.now() - start) / 1000), 100)
    }
    return () => clearInterval(clockRef.current)
  }, [taskId, poll?.result, poll?.state])

  // Poll the backend
  useEffect(() => {
    if (!taskId) return
    timerRef.current = setInterval(async () => {
      try {
        const data = await getInspection(taskId)
        setPoll(data)

        // Reflect live "completed" list from the worker while it runs
        if (data.state === 'PROGRESS' && data.progress?.completed) {
          setAgentStatus((s) => {
            const next = { ...s }
            data.progress.completed.forEach((k) => (next[k] = 'done'))
            if (data.progress.current) next[data.progress.current] = 'running'
            return next
          })
        }
        if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
          clearInterval(timerRef.current)
        }
      } catch (e) {
        setError(e.message)
        clearInterval(timerRef.current)
      }
    }, 700)
    return () => clearInterval(timerRef.current)
  }, [taskId])

  // Once the run finishes, cascade the agents so you SEE them work.
  useEffect(() => {
    const result = poll?.result
    if (!result || revealedFor.current === taskId) return
    revealedFor.current = taskId
    ;(async () => {
      setRevealing(true)
      for (const a of AGENTS) {
        const target = finalStatusFor(a.key, result)
        if (target === 'skipped') {
          setAgentStatus((s) => ({ ...s, [a.key]: 'skipped' }))
          await sleep(90)
          continue
        }
        setAgentStatus((s) => ({ ...s, [a.key]: 'running' }))
        await sleep(480)
        setAgentStatus((s) => ({ ...s, [a.key]: target }))
        await sleep(160)
      }
      setRevealing(false)
    })()
  }, [poll?.result, taskId])

  const setInspField = (k) => (e) => setInsp({ ...insp, [k]: e.target.value })
  const setSensorField = (k) => (e) =>
    setSensor({ ...sensor, [k]: k === 'anomaly_flagged' ? e.target.checked : e.target.value })

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setPoll(null)
    setTaskId(null)
    setElapsed(0)
    setAgentStatus(initStatus())
    revealedFor.current = null
    try {
      const sensors = [
        {
          sensor_id: sensor.sensor_id,
          timestamp: insp.timestamp,
          temperature_c: Number(sensor.temperature_c),
          machine_speed_rpm: Number(sensor.machine_speed_rpm),
          anomaly_flagged: Boolean(sensor.anomaly_flagged),
        },
      ]
      // Mark first agent running immediately for instant feedback
      setAgentStatus((s) => ({ ...s, ingestor: 'running' }))
      const { task_id } = await submitInspection(insp, sensors)
      setTaskId(task_id)
    } catch (e) {
      setError(e.message)
      setAgentStatus(initStatus())
    }
  }

  const result = poll?.result
  const done = poll?.state === 'SUCCESS' || poll?.state === 'FAILURE'
  const running = taskId && (!done || revealing)
  const doneCount = Object.values(agentStatus).filter((s) => s === 'done').length
  const singleProgress = doneCount / AGENTS.length

  return (
    <div className="page">
      <header className="app-header">
        <div>
          <h1>ManufactureIQ</h1>
          <p className="subtitle">Multi-agent defect inspection pipeline</p>
        </div>
        <div className="header-stats">
          <Stat label="Agents" value={`${doneCount}/${AGENTS.length}`} />
          <Stat label="Elapsed" value={`${elapsed.toFixed(1)}s`} />
          <RunPill running={running} state={poll?.state} status={result?.status} />
        </div>
      </header>

      <div className="tabs">
        <button className={`tab ${mode === 'single' ? 'active' : ''}`} onClick={() => setMode('single')}>Single inspection</button>
        <button className={`tab ${mode === 'batch' ? 'active' : ''}`} onClick={() => setMode('batch')}>Batch (CSV)</button>
      </div>

      {mode === 'batch' ? (
        <BatchPanel />
      ) : (
      <div className="layout">
        {/* ── Sidebar: inspection form ─────────────────────── */}
        <form className="card sidebar" onSubmit={handleSubmit}>
          <div className="card-head">
            <h2>Inspection</h2>
            <button type="button" className="link" onClick={() => { setInsp(SAMPLE_INSPECTION); setSensor(SAMPLE_SENSOR) }}>
              Reset sample
            </button>
          </div>
          <div className="fields">
            <Field label="Record ID"  value={insp.record_id}  onChange={setInspField('record_id')} />
            <Field label="Batch ID"   value={insp.batch_id}   onChange={setInspField('batch_id')} />
            <Field label="Product ID" value={insp.product_id} onChange={setInspField('product_id')} />
            <Field label="Machine ID" value={insp.machine_id} onChange={setInspField('machine_id')} />
          </div>
          <Field label="Timestamp" value={insp.timestamp} onChange={setInspField('timestamp')} full />
          <label className="notes-label">Notes
            <textarea rows={2} value={insp.raw_notes} onChange={setInspField('raw_notes')} />
          </label>

          <h3>Sensor reading</h3>
          <Field label="Sensor ID" value={sensor.sensor_id} onChange={setSensorField('sensor_id')} full />
          <div className="fields">
            <Field label="Temp °C"   value={sensor.temperature_c}     onChange={setSensorField('temperature_c')} type="number" />
            <Field label="Speed RPM" value={sensor.machine_speed_rpm} onChange={setSensorField('machine_speed_rpm')} type="number" />
          </div>
          <label className="check">
            <input type="checkbox" checked={sensor.anomaly_flagged} onChange={setSensorField('anomaly_flagged')} />
            Anomaly flagged
          </label>

          <button className="submit" type="submit" disabled={running}>
            {running ? 'Running pipeline…' : 'Run inspection'}
          </button>
          {error && <div className="alert error">{error}</div>}
        </form>

        {/* ── Main: pipeline + outputs ─────────────────────── */}
        <main className="main">
          <AgentPipeline agentStatus={agentStatus} tiers={TIERS} agents={AGENTS} progress={singleProgress} />

          {result && result.sensor_summary && (
            <Telemetry summary={result.sensor_summary} />
          )}

          {!taskId && (
            <div className="card placeholder">
              <p>Submit an inspection to watch the agents process it.</p>
            </div>
          )}

          {result && (
            <AgentDetails state={result} agentStatus={agentStatus} />
          )}
        </main>
      </div>
      )}
    </div>
  )
}

function Field({ label, value, onChange, type = 'text', full }) {
  return (
    <label className={`field ${full ? 'full' : ''}`}>
      <span>{label}</span>
      <input type={type} value={value} onChange={onChange} />
    </label>
  )
}

function Stat({ label, value }) {
  return (
    <div className="hstat">
      <span className="hstat-value">{value}</span>
      <span className="hstat-label">{label}</span>
    </div>
  )
}

function RunPill({ running, state, status }) {
  let cls = 'idle', text = 'Idle'
  if (running) { cls = 'run'; text = 'Processing' }
  else if (status === 'failed' || state === 'FAILURE') { cls = 'fail'; text = 'Failed' }
  else if (state === 'SUCCESS') { cls = 'ok'; text = 'Complete' }
  return <span className={`runpill ${cls}`}>{running && <span className="dot" />}{text}</span>
}
