import { useEffect, useRef } from 'react'

// Draws a WM-811K wafer map: 0 = no die, 1 = passing die, 2 = failing die.
// Colours come from CSS variables so the map follows light/dark theme.
export default function WaferMap({ grid, size = 220, caption }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !grid?.length) return
    const h = grid.length
    const w = grid[0].length
    const dpr = window.devicePixelRatio || 1
    const cell = Math.max(1, Math.floor(size / Math.max(h, w)))
    canvas.width = w * cell * dpr
    canvas.height = h * cell * dpr
    canvas.style.width = `${w * cell}px`
    canvas.style.height = `${h * cell}px`

    const css = getComputedStyle(canvas)
    const pass = css.getPropertyValue('--wafer-pass').trim() || '#cbd5e1'
    const fail = css.getPropertyValue('--wafer-fail').trim() || '#dc2626'
    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w * cell, h * cell)
    const gap = cell >= 4 ? 0.5 : 0
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = grid[y][x]
        if (!v) continue
        ctx.fillStyle = v === 2 ? fail : pass
        ctx.fillRect(x * cell, y * cell, cell - gap, cell - gap)
      }
    }
  }, [grid, size])

  if (!grid?.length) return null
  return (
    <figure className="wafer">
      <canvas ref={ref} role="img"
        aria-label={`Wafer map ${grid.length} by ${grid[0].length} die`} />
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  )
}
