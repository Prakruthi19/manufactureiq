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
