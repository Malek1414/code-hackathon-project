import type { ProjectedBox } from './Viewer'

export interface Callout { label: string; value: string }

/** Edge list for the projected bounding box, in Viewer's CORNER_ORDER. */
const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
]

const SHELF = 84
const ROW = 38

function polylineLength(pts: { x: number; y: number }[]) {
  let n = 0
  for (let i = 1; i < pts.length; i++) n += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
  return Math.ceil(n)
}

/**
 * The atlas's signature: CAD leader lines that draw themselves onto the model.
 * Anchors come from the projected bounding box, so they track the part as it
 * orbits and explodes rather than sitting in a fixed caption.
 */
export function Callouts({
  box, callouts, seed, width, height,
}: {
  box: ProjectedBox | null
  callouts: Callout[]
  /** changing this replays the draw-on animation */
  seed: string
  width: number
  height: number
}) {
  if (!box || !box.visible || !callouts.length) return null

  const xs = box.corners.map((c) => c.x)
  const ys = box.corners.map((c) => c.y)
  const maxX = Math.max(...xs)
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)

  // Flip the whole callout stack to the left when it would run off the sheet.
  // leave room for the longest label, not just the shelf
  const flip = maxX + SHELF + 150 > width
  const dir = flip ? -1 : 1
  const originX = flip ? minX : maxX

  // Rightmost (or leftmost when flipped) corners make the most readable anchors,
  // because a leader that crosses the body reads as a line through the part.
  const anchors = [...box.corners].sort((a, b) => (flip ? a.x - b.x : b.x - a.x))

  const rows = callouts.slice(0, 3)
  const topY = Math.max(24, Math.min(minY - 4, height - rows.length * ROW - 24))

  return (
    <svg className="overlay" width={width} height={height} key={seed} aria-hidden="true">
      <g className="fade-in">
        {EDGES.map(([a, b], i) => (
          <line
            key={i}
            className="callout-box"
            x1={box.corners[a].x} y1={box.corners[a].y}
            x2={box.corners[b].x} y2={box.corners[b].y}
          />
        ))}
      </g>

      {rows.map((c, i) => {
        const anchor = anchors[i % anchors.length]
        const shelfY = topY + i * ROW
        const elbowX = originX + dir * (SHELF - 26)
        const endX = originX + dir * (SHELF + 38)
        const pts = [anchor, { x: elbowX, y: shelfY }, { x: endX, y: shelfY }]
        const len = polylineLength(pts)
        const textX = flip ? endX + 6 : endX - 6
        const anchorAttr = flip ? 'start' : 'end'

        return (
          <g key={c.label}>
            <circle className="callout-tick" cx={anchor.x} cy={anchor.y} r={2} fill="var(--vermilion)" />
            <polyline
              className="callout-line draw-on"
              points={pts.map((p) => `${p.x},${p.y}`).join(' ')}
              style={{ ['--len' as string]: len, animationDelay: `${i * 70}ms` }}
            />
            <text className="callout-text fade-in" x={textX} y={shelfY - 5} textAnchor={anchorAttr}
                  style={{ animationDelay: `${220 + i * 70}ms` }}>
              {c.value}
            </text>
            <text className="callout-label fade-in" x={textX} y={shelfY + 11} textAnchor={anchorAttr}
                  style={{ animationDelay: `${260 + i * 70}ms` }}>
              {c.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
