// Reusable determinate progress bar.

export default function ProgressBar({ value = 0, label, tone = 'accent' }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)))
  return (
    <div className="progress">
      {label && (
        <div className="progress-head">
          <span>{label}</span>
          <span className="progress-pct">{pct}%</span>
        </div>
      )}
      <div className="progress-track">
        <div className={`progress-fill tone-${tone}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}
