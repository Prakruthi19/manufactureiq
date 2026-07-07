// CSV parsing + row → pipeline-payload mapping for batch inspection uploads.
// Designed to accept the AI4I 2020 dataset columns directly, but tolerant of
// a few header spellings.

export const BATCH_LIMIT = 100

const num = (v) => {
  const n = parseFloat(v)
  return Number.isFinite(n) ? n : null
}

// Minimal CSV splitter (the AI4I dataset has no quoted commas).
function splitLine(line) {
  return line.split(',')
}

export function parseCsv(text) {
  const lines = text.replace(/﻿/g, '').trim().split(/\r?\n/)
  if (lines.length < 2) return []
  const headers = splitLine(lines[0]).map((h) => h.trim())
  return lines.slice(1).filter(Boolean).map((line) => {
    const cells = splitLine(line)
    const row = {}
    headers.forEach((h, i) => (row[h] = (cells[i] ?? '').trim()))
    return row
  })
}

// Try several header spellings for one logical field.
const pick = (row, keys) => {
  for (const k of keys) if (row[k] !== undefined && row[k] !== '') return row[k]
  return undefined
}

export function rowToPayload(row, idx) {
  const airK = pick(row, ['Air temperature [K]', 'Air temperature (K)', 'air_temperature_k'])
  const procK = pick(row, ['Process temperature [K]', 'Process temperature (K)', 'process_temperature_k'])
  const rpm = pick(row, ['Rotational speed [rpm]', 'Rotational speed (rpm)', 'rpm'])
  const torque = pick(row, ['Torque [Nm]', 'Torque (Nm)', 'torque_nm'])
  const wear = pick(row, ['Tool wear [min]', 'Tool wear (min)', 'tool_wear_min'])
  const pid = pick(row, ['Product ID', 'product_id']) || `P${idx + 1}`
  const type = pick(row, ['Type', 'type']) || pid[0] || 'M'
  const udi = pick(row, ['UDI', 'udi']) || idx + 1

  const inspection = {
    record_id: `REC-${udi}`,
    batch_id: 'BATCH-CSV',
    product_id: pid,
    machine_id: `MCH-${type}`,
    timestamp: '2026-07-06T09:00:00',
    raw_notes: '',
    air_temperature_k: num(airK),
    process_temperature_k: num(procK),
    torque_nm: num(torque),
    tool_wear_min: num(wear),
  }
  const sensors = [
    {
      sensor_id: `${pid}-S1`,
      timestamp: '2026-07-06T09:00:00',
      temperature_c: airK !== undefined ? num(airK) - 273.15 : null,
      machine_speed_rpm: num(rpm),
      anomaly_flagged: false,
    },
  ]
  return { inspection, sensors, label: inspection.record_id, product: pid }
}
