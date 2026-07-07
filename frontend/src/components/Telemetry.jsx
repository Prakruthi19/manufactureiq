// Sensor-derived metrics + risk flags produced by the Sensor Parser agent.

const RISKS = [
  { key: 'hdf_risk', label: 'HDF', title: 'Heat Dissipation Failure' },
  { key: 'pwf_risk', label: 'PWF', title: 'Power Failure' },
  { key: 'twf_risk', label: 'TWF', title: 'Tool Wear Failure' },
  { key: 'osf_risk', label: 'OSF', title: 'Overstrain Failure' },
]

const fmt = (v, unit = '') =>
  v === null || v === undefined ? '—' : `${typeof v === 'number' ? Math.round(v * 100) / 100 : v}${unit}`

export default function Telemetry({ summary }) {
  return (
    <div className="card telemetry">
      <div className="card-head"><h2>Sensor telemetry</h2>
        <span className={`chip ${summary.any_anomaly ? 'chip-warn' : 'chip-ok'}`}>
          {summary.any_anomaly ? `${summary.anomaly_count} anomalies` : 'nominal'}
        </span>
      </div>
      <div className="tiles">
        <Tile label="Power" value={fmt(summary.power_w, ' W')} />
        <Tile label="Temp Δ" value={fmt(summary.temp_diff_k, ' K')} />
        <Tile label="Wear·Torque" value={fmt(summary.wear_torque)} />
        <Tile label="Anomalies" value={fmt(summary.anomaly_count)} />
      </div>
      <div className="risk-row">
        {RISKS.map((r) => (
          <span key={r.key} title={r.title} className={`risk ${summary[r.key] ? 'risk-on' : 'risk-off'}`}>
            {r.label}
          </span>
        ))}
      </div>
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
