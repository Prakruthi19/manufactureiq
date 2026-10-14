// Thin client for the ManufactureIQ FastAPI backend.
// Requests go to /api/* and are proxied to :8000 by vite.config.js.

export async function submitInspection(inspection, sensors, full = true) {
  const res = await fetch('/api/inspections', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inspection, sensors, full }),
  })
  if (!res.ok) throw new Error(`Submit failed (${res.status})`)
  return res.json() // { task_id, status }
}

export async function getInspection(taskId) {
  const res = await fetch(`/api/inspections/${taskId}`)
  if (!res.ok) throw new Error(`Fetch failed (${res.status})`)
  return res.json() // { task_id, state, progress?, result?, error? }
}

export async function getSamples() {
  const res = await fetch('/api/samples')
  if (!res.ok) throw new Error(`Could not load demo wafers (${res.status}) — run scripts/build_demo_samples.py`)
  return res.json() // { singles: [...], scenarios: [...], note }
}

export async function getModelCard() {
  const res = await fetch('/api/model-card')
  if (!res.ok) throw new Error(`Could not load model card (${res.status})`)
  return res.json()
}

export async function clearHistory() {
  const res = await fetch('/api/history', { method: 'DELETE' })
  if (!res.ok) throw new Error(`Could not clear history (${res.status})`)
  return res.json()
}

export async function getDrift() {
  const res = await fetch('/api/drift')
  if (!res.ok) throw new Error(`Could not load drift report (${res.status})`)
  return res.json()
}
