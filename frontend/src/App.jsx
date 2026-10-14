import { useEffect, useRef, useState } from 'react'
import { submitInspection, getInspection, getSamples } from './api'
import { AGENTS, TIERS, finalStatusFor, sleep } from './agents'
import AgentPipeline from './components/AgentPipeline'
import AgentDetails from './components/AgentDetails'
import BatchPanel from './components/BatchPanel'
import ModelCard from './components/ModelCard'
import WaferMap from './components/WaferMap'
import Prediction from './components/Prediction'

// Every demo wafer is a HELD-OUT WM-811K wafer (from lots the model never saw),
// paired with a held-out SECOM process run. The true expert label stays hidden
// until the model has answered, then the UI shows predicted vs actual.

const initStatus = () => Object.fromEntries(AGENTS.map((a) => [a.key, 'queued']))

export default function App() {
  const [mode, setMode] = useState('single') // 'single' | 'batch' | 'model'
  const [samples, setSamples] = useState(null)
  const [sampleIdx, setSampleIdx] = useState(0)
  const [insp, setInsp] = useState(null)
  const [taskId, setTaskId] = useState(null)
  const [poll, setPoll] = useState(null)
  const [error, setError] = useState(null)
  const [agentStatus, setAgentStatus] = useState(initStatus())
  const [revealing, setRevealing] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  const timerRef = useRef(null)
  const clockRef = useRef(null)
  const revealedFor = useRef(null)
  const sawLive = useRef(false)

  useEffect(() => {
    getSamples()
      .then((s) => { setSamples(s); setInsp(s.singles[0].inspection) })
      .catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    const active = taskId && !poll?.result && poll?.state !== 'FAILURE'
    if (active) {
      const start = Date.now()
      clockRef.current = setInterval(() => setElapsed((Date.now() - start) / 1000), 100)
    }
    return () => clearInterval(clockRef.current)
  }, [taskId, poll?.result, poll?.state])

  useEffect(() => {
    if (!taskId) return
    timerRef.current = setInterval(async () => {
      try {
        const data = await getInspection(taskId)
        setPoll(data)
        if (data.state === 'PROGRESS' && data.progress?.completed) {
          if (data.progress.completed.length) sawLive.current = true
          setAgentStatus((s) => {
            const next = { ...s }
            data.progress.completed.forEach((k) => (next[k] = 'done'))
            if (data.progress.current) next[data.progress.current] = 'running'
            return next
          })
        }
        if (data.state === 'SUCCESS' || data.state === 'FAILURE') clearInterval(timerRef.current)
      } catch (e) {
        setError(e.message)
        clearInterval(timerRef.current)
      }
    }, 700)
    return () => clearInterval(timerRef.current)
  }, [taskId])

  // Settle agents on their final status (cascade only if no live progress was seen).
  useEffect(() => {
    const result = poll?.result
    if (!result || revealedFor.current === taskId) return
    revealedFor.current = taskId
    if (sawLive.current) {
      setAgentStatus(Object.fromEntries(AGENTS.map((a) => [a.key, finalStatusFor(a.key, result)])))
      return
    }
    ;(async () => {
      setRevealing(true)
      for (const a of AGENTS) {
        const target = finalStatusFor(a.key, result)
        if (target === 'skipped') { setAgentStatus((s) => ({ ...s, [a.key]: 'skipped' })); await sleep(90); continue }
        setAgentStatus((s) => ({ ...s, [a.key]: 'running' }))
        await sleep(420)
        setAgentStatus((s) => ({ ...s, [a.key]: target }))
        await sleep(140)
      }
      setRevealing(false)
    })()
  }, [poll?.result, taskId])

  const sample = samples?.singles[sampleIdx]
  const loadSample = (i) => {
    setSampleIdx(i); setInsp(samples.singles[i].inspection)
    setTaskId(null); setPoll(null); setAgentStatus(initStatus())
  }
  const setField = (k) => (e) => setInsp({ ...insp, [k]: e.target.value })

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null); setPoll(null); setTaskId(null); setElapsed(0)
    setAgentStatus(initStatus())
    revealedFor.current = null
    sawLive.current = false
    try {
      setAgentStatus((s) => ({ ...s, ingestor: 'running' }))
      const { task_id } = await submitInspection(insp, [], true)
      setTaskId(task_id)
    } catch (e) {
      setError(e.message)
      setAgentStatus(initStatus())
    }
  }

  const result = poll?.result
  const done = poll?.state === 'SUCCESS' || poll?.state === 'FAILURE'
  const running = taskId && (!done || revealing)
  const healthy = done && !revealing && result?.classification?.defect_type === 'no_defect'
  const doneCount = Object.values(agentStatus).filter((s) => s === 'done').length

  return (
    <div className="page">
      <header className="app-header">
        <div>
          <h1>ManufactureIQ</h1>
          <p className="subtitle">Semiconductor wafer inspection · 7-agent pipeline · trained on WM-811K + SECOM</p>
        </div>
        <div className="header-stats">
          <Stat label="Agents" value={`${doneCount}/${AGENTS.length}`} />
          <Stat label="Elapsed" value={`${elapsed.toFixed(1)}s`} />
          <RunPill running={running} state={poll?.state} status={result?.status} />
        </div>
      </header>

      <div className="tabs">
        <button className={`tab ${mode === 'single' ? 'active' : ''}`} onClick={() => setMode('single')}>Single wafer</button>
        <button className={`tab ${mode === 'batch' ? 'active' : ''}`} onClick={() => setMode('batch')}>Lot scenarios</button>
        <button className={`tab ${mode === 'model' ? 'active' : ''}`} onClick={() => setMode('model')}>Model card</button>
      </div>

      {mode === 'model' && <ModelCard />}
      {mode === 'batch' && <BatchPanel scenarios={samples?.scenarios} note={samples?.note} />}
      {mode === 'single' && (
      <div className="layout">
        <form className="card sidebar" onSubmit={handleSubmit}>
          <div className="card-head"><h2>Wafer inspection</h2></div>
          {!samples && !error && <p className="batch-note">Loading held-out wafers…</p>}
          {samples && insp && (<>
            <label className="field full"><span>Held-out wafer</span>
              <select className="sample-select" value={sampleIdx} onChange={(e) => loadSample(Number(e.target.value))}>
                {samples.singles.map((s, i) => <option key={i} value={i}>{s.label}</option>)}
              </select>
            </label>
            <WaferMap grid={insp.wafer_map} size={240}
              caption={`${insp.wafer_map.length}×${insp.wafer_map[0].length} grid · red = failing die`} />
            <div className="fields">
              <Field label="Lot" value={insp.batch_id} onChange={setField('batch_id')} />
              <Field label="Tool" value={insp.machine_id} onChange={setField('machine_id')} />
              <Field label="Supplier" value={insp.supplier_id || ''} onChange={setField('supplier_id')} />
              <Field label="Material lot" value={insp.material_batch_id || ''} onChange={setField('material_batch_id')} />
            </div>
            <p className="batch-hint">
              Process run: SECOM row {sample.secom_run.row} (held-out period, 590 sensors).
              Tool / supplier IDs are simulated — the datasets don't include them.
            </p>
            <button className="submit" type="submit" disabled={running}>
              {running ? 'Running pipeline…' : 'Run inspection'}
            </button>
          </>)}
          {error && <div className="alert error">{error}</div>}
        </form>

        <main className="main">
          {healthy && (
            <div className="alert ok">
              <strong>✓ No defect pattern.</strong> Root cause, supplier alert and compliance are
              skipped for clean wafers by design. The trend detector still recorded it in the lot history.
            </div>
          )}
          <AgentPipeline agentStatus={agentStatus} tiers={TIERS} agents={AGENTS} progress={doneCount / AGENTS.length} />
          {!taskId && <div className="card placeholder"><p>Pick a held-out wafer and run it. The expert label is revealed after the model answers.</p></div>}
          {result?.classification && !revealing && <Prediction state={result} truth={sample?.true_pattern} secom={sample?.secom_run} />}
          {result && <AgentDetails state={result} agentStatus={agentStatus} />}
        </main>
      </div>
      )}
    </div>
  )
}

function Field({ label, value, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input value={value} onChange={onChange} />
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
