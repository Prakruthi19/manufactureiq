import { useState } from 'react'

// Reliability diagram: for each confidence bucket, how often the model was
// actually right. A perfectly honest model sits on the diagonal.
// Two series (before / after temperature scaling) + the dashed "perfect" line.

const W = 360, H = 260, M = { top: 12, right: 12, bottom: 40, left: 46 }
const IW = W - M.left - M.right, IH = H - M.top - M.bottom
const LO = 0.3                                   // no wafer is below ~0.3 confidence (9 classes)
const x = (v) => M.left + ((v - LO) / (1 - LO)) * IW
const y = (v) => M.top + (1 - v) * IH
const TICKS = [0.3, 0.5, 0.7, 0.9, 1.0]
const YTICKS = [0, 0.25, 0.5, 0.75, 1]

const SERIES = [
  { key: 'bins_before', label: 'Before calibration', cls: 'rel-before' },
  { key: 'bins_after', label: 'After calibration', cls: 'rel-after' },
]

export default function ReliabilityChart({ calibration }) {
  const [hover, setHover] = useState(null)
  const pts = (bins) => bins.filter((b) => b.confidence >= LO)

  return (
    <div className="rel-wrap viz-root">
      <div className="rel-legend" aria-hidden="true">
        {SERIES.map((s) => <span key={s.key}><i className={`rel-swatch ${s.cls}`} />{s.label}</span>)}
        <span><i className="rel-swatch rel-ideal" />Perfectly honest</span>
      </div>
      <div className="rel-chart">
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label="Reliability diagram: share of predictions that were correct, by model confidence, before and after calibration">
          {YTICKS.map((t) => (
            <g key={t}>
              <line className="rel-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
              <text className="rel-tick" x={M.left - 8} y={y(t) + 4} textAnchor="end">{Math.round(t * 100)}%</text>
            </g>
          ))}
          {TICKS.map((t) => (
            <text key={t} className="rel-tick" x={x(t)} y={H - M.bottom + 16} textAnchor="middle">{Math.round(t * 100)}%</text>
          ))}
          <text className="rel-axis" x={M.left + IW / 2} y={H - 4} textAnchor="middle">Model confidence</text>
          <text className="rel-axis" transform={`translate(12 ${M.top + IH / 2}) rotate(-90)`} textAnchor="middle">Actually correct</text>

          <line className="rel-ideal-line" x1={x(LO)} y1={y(LO)} x2={x(1)} y2={y(1)} />

          {SERIES.map((s) => {
            const p = pts(calibration[s.key])
            return (
              <g key={s.key} className={s.cls}>
                <polyline className="rel-line" points={p.map((b) => `${x(b.confidence)},${y(b.accuracy)}`).join(' ')} />
                {p.map((b, i) => (
                  <g key={i}>
                    <circle className="rel-dot" cx={x(b.confidence)} cy={y(b.accuracy)} r={hover?.s === s.key && hover.i === i ? 6 : 4} />
                    <circle className="rel-hit" cx={x(b.confidence)} cy={y(b.accuracy)} r={12}
                      onMouseEnter={() => setHover({ s: s.key, i, b, label: s.label })}
                      onMouseLeave={() => setHover(null)} />
                  </g>
                ))}
              </g>
            )
          })}
        </svg>
        {hover && (
          <div className="rel-tip" style={{
            left: `${(x(hover.b.confidence) / W) * 100}%`, top: `${(y(hover.b.accuracy) / H) * 100}%` }}>
            <strong>{hover.label}</strong>
            <span>Confidence {(hover.b.confidence * 100).toFixed(1)}%</span>
            <span>Correct {(hover.b.accuracy * 100).toFixed(1)}%</span>
            <span>{hover.b.n.toLocaleString()} wafers</span>
          </div>
        )}
      </div>
    </div>
  )
}
