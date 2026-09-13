import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Viewer, type GeometryRecord, type ProjectedBox } from './viewer/Viewer'
import { Callouts, type Callout } from './viewer/Callouts'
import type { SystemId } from './data/parts'
import type { Device } from './data/devices'

const round = (n: number) => (n >= 100 ? n.toFixed(0) : n.toFixed(1))

const FLICKER_COLOR: Record<string, string> = {
  source: '#d8341a',
  lever: '#0f4c5c',
  path: '#7fa3b5',
}
const FLICKER_LABEL: Record<string, string> = {
  source: 'Emits or modulates light — can flicker',
  lever: 'Where you can intervene',
  path: 'Light passes through — shapes what arrives',
}

export function DeviceAtlas({ device }: { device: Device }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<Viewer | null>(null)

  const GEO_BY_ID = useMemo(() => new Map(device.geometry.map((g) => [g.id, g])), [device])
  const COLOR = useMemo(
    () => Object.fromEntries(device.systems.map((s) => [s.id, s.color])) as Record<string, string>,
    [device],
  )
  const GHOST = useMemo(
    () => new Set(device.systems.filter((s) => s.ghost).map((s) => s.id)),
    [device],
  )

  const [ready, setReady] = useState(false)
  const [systems, setSystems] = useState<Set<SystemId>>(() => new Set(device.defaultSystems))
  const [only, setOnly] = useState<string[] | null>(null)
  const [preset, setPreset] = useState<string | null>(device.presets[0]?.id ?? null)
  const [step, setStep] = useState<number | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [isolated, setIsolated] = useState(false)
  const [flickerMode, setFlickerMode] = useState(false)
  const [explode, setExplode] = useState(0)
  const [query, setQuery] = useState('')
  const [box, setBox] = useState<ProjectedBox | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  const hasFlicker = useMemo(() => device.parts.some((p) => p.flicker), [device])

  const visibleIds = useMemo(() => {
    const set = new Set<string>()
    for (const g of device.geometry) {
      if (!systems.has(g.system as SystemId)) continue
      if (only && !only.includes(g.id)) continue
      if (!only && g.defaultHidden) continue
      set.add(g.id)
    }
    return set
  }, [systems, only, device])

  const accent = useMemo(() => {
    if (!flickerMode) return null
    const map: Record<string, string> = {}
    for (const g of device.geometry) {
      const role = device.byId[g.id]?.flicker?.role
      map[g.id] = role ? FLICKER_COLOR[role] : '#c4cdd2'
    }
    return map
  }, [flickerMode, device])

  useEffect(() => {
    const host = stageRef.current
    if (!host) return
    const viewer = new Viewer(host)
    viewerRef.current = viewer
    viewer.onPick = (id) => { setSelected(id); setPreset(null) }
    viewer.onProject = setBox

    let cancelled = false
    viewer.load(
      device.model,
      device.geometry as GeometryRecord[],
      (s) => COLOR[s] ?? '#7E8C97',
      (s) => GHOST.has(s as SystemId),
    )
      .then(() => { if (!cancelled) setReady(true) })
      .catch((e) => console.error('[atlas] model failed to load', e))

    const ro = new ResizeObserver(() => {
      viewer.resize()
      setSize({ w: host.clientWidth, h: host.clientHeight })
    })
    ro.observe(host)
    setSize({ w: host.clientWidth, h: host.clientHeight })

    return () => { cancelled = true; ro.disconnect(); viewer.dispose(); viewerRef.current = null }
  }, [device, COLOR, GHOST])

  useEffect(() => { if (ready) viewerRef.current?.setVisible(visibleIds) }, [visibleIds, ready])
  useEffect(() => { viewerRef.current?.setExplode(explode) }, [explode])
  useEffect(() => { viewerRef.current?.setSelected(selected) }, [selected])
  useEffect(() => { viewerRef.current?.setIsolated(isolated) }, [isolated])
  useEffect(() => { if (ready) viewerRef.current?.setAccent(accent) }, [accent, ready])

  // Re-frame after the visible set or the explode target changes, so a preset that
  // spreads parts across half a metre still arrives in shot.
  useEffect(() => {
    if (!ready) return
    const t = setTimeout(() => viewerRef.current?.reframe(), 0)
    return () => clearTimeout(t)
  }, [visibleIds, explode, isolated, ready])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setSelected(null); setIsolated(false); setQuery('') }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const applyPreset = useCallback((id: string) => {
    const p = device.presets.find((x) => x.id === id)
    if (!p) return
    setPreset(id); setStep(null); setOnly(p.only ?? null)
    setSystems(new Set(p.systems)); setExplode(p.explode ?? 0)
    setIsolated(false); setSelected(null)
    if (p.id === 'flicker') setFlickerMode(true)
  }, [device])

  const applyStep = useCallback((n: number) => {
    const s = device.assembly?.find((x) => x.n === n)
    if (!s) return
    setStep(n); setPreset(null); setOnly(s.parts)
    setSystems(new Set(device.systems.map((x) => x.id)))
    setExplode(s.explode); setIsolated(false); setSelected(s.parts[0] ?? null)
  }, [device])

  const toggleSystem = useCallback((id: SystemId) => {
    setPreset(null); setStep(null); setOnly(null)
    setSystems((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])

  const pick = useCallback((id: string) => {
    const g = GEO_BY_ID.get(id)
    if (g && (!systems.has(g.system as SystemId) || (only && !only.includes(id)))) {
      setOnly(null); setPreset(null); setStep(null)
      setSystems((prev) => new Set(prev).add(g.system as SystemId))
    }
    setSelected(id); setQuery('')
  }, [systems, only, GEO_BY_ID])

  const hits = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return []
    return device.parts.filter((p) => {
      const g = GEO_BY_ID.get(p.id)
      return [p.name, p.id, p.code, p.role, p.does, p.flicker?.note, g?.object, g?.system]
        .filter(Boolean).join(' ').toLowerCase().includes(q)
    }).slice(0, 10)
  }, [query, device, GEO_BY_ID])

  const counts = useMemo(() => {
    const m: Record<string, number> = {}
    for (const g of device.geometry) m[g.system] = (m[g.system] ?? 0) + 1
    return m
  }, [device])

  const part = selected ? device.byId[selected] : null
  const geo = selected ? GEO_BY_ID.get(selected) : null

  const callouts: Callout[] = useMemo(() => {
    if (!part || !geo) return []
    if (part.callouts?.length) return part.callouts
    if (device.schematic) return []   // never draw a dimension line on invented geometry
    const [x, y, z] = geo.sizeMm
    return [{ label: 'Overall', value: `${round(x)} × ${round(y)} × ${round(z)}` }]
  }, [part, geo, device])

  const printedCount = device.geometry.filter((g) => g.printed).length

  return (
    <>
      <div className="subhead">
        <p className="subhead-role">{device.role}</p>
        <div className="search">
          <svg className="search-icon" width="11" height="11" viewBox="0 0 12 12" fill="none">
            <circle cx="5" cy="5" r="3.6" stroke="currentColor" strokeWidth="1.2" />
            <path d="M7.8 7.8L11 11" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
          <input value={query} onChange={(e) => setQuery(e.target.value)}
                 placeholder="find a part, a number, a job" aria-label="Search parts" />
          {query.trim() !== '' && (
            <div className="results">
              {hits.length === 0 && <div className="results-empty">Nothing matches “{query}”.</div>}
              {hits.map((p) => (
                <button key={p.id} className="result" onClick={() => pick(p.id)}>
                  <span className="result-code">{p.code ?? '—'}</span>
                  <span className="result-name">{p.name}</span>
                  <span className="result-sys">{GEO_BY_ID.get(p.id)?.system}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid">
        <aside className="rail">
          <div className="block">
            <h2 className="block-head">
              Systems · {device.geometry.length} parts{printedCount ? ` · ${printedCount} printed` : ''}
            </h2>
            {device.systems.map((s) => {
              const on = systems.has(s.id)
              return (
                <button key={s.id} className="layer" data-on={on} aria-pressed={on}
                        onClick={() => toggleSystem(s.id)}>
                  <span className="swatch" style={{ background: s.color }} />
                  <span>
                    <span className="layer-name">{s.name}</span>
                    <span className="layer-note">{s.note}</span>
                  </span>
                  <span className="layer-count">{counts[s.id] ?? 0}</span>
                </button>
              )
            })}
          </div>

          <div className="block">
            <h2 className="block-head">Views</h2>
            {device.presets.map((p) => (
              <button key={p.id} className="preset" data-on={preset === p.id}
                      onClick={() => applyPreset(p.id)}>
                <span className="preset-name">{p.name}</span>
                <span className="preset-q">{p.question}</span>
              </button>
            ))}
          </div>

          {hasFlicker && (
            <div className="block">
              <h2 className="block-head">Photosensitivity</h2>
              <button className="preset" data-on={flickerMode} onClick={() => setFlickerMode((v) => !v)}
                      aria-pressed={flickerMode}>
                <span className="preset-name">Colour by flicker role</span>
                <span className="preset-q">Which parts put light in the eye, and which ones you control.</span>
              </button>
              {flickerMode && (
                <div className="legend">
                  {(['source', 'lever', 'path'] as const).map((k) => (
                    <span key={k} className="legend-row">
                      <span className="swatch" style={{ background: FLICKER_COLOR[k] }} />
                      {FLICKER_LABEL[k]}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {device.assembly && (
            <div className="block">
              <h2 className="block-head">Assembly order</h2>
              {device.assembly.map((s) => (
                <button key={s.n} className="step" data-on={step === s.n} onClick={() => applyStep(s.n)}>
                  <span className="step-n">{String(s.n).padStart(2, '0')}</span>
                  <span className="step-t">{s.text}</span>
                </button>
              ))}
            </div>
          )}

          <div className="block">
            <h2 className="block-head">Where this model comes from</h2>
            <p className="note" data-kind={device.schematic ? 'watch' : 'source'}>{device.provenance}</p>
          </div>
        </aside>

        <section className="stage">
          <div className="canvas" ref={stageRef} />
          <Callouts box={box} callouts={callouts} seed={selected ?? 'none'} width={size.w} height={size.h} />
          {!ready && <div className="loading">loading geometry…</div>}
          {ready && !selected && <div className="stage-hint">drag to orbit · click any part</div>}
          {device.schematic && <div className="stage-badge">block schematic</div>}

          <div className="controls">
            <span className="controls-label">Explode</span>
            <input type="range" min={0} max={1} step={0.01} value={explode}
                   onChange={(e) => { setExplode(Number(e.target.value)); setPreset(null) }}
                   aria-label="Explode the assembly" />
            <span className="controls-val">{Math.round(explode * 100)}%</span>
            <button className="pill" data-on={isolated} disabled={!selected}
                    onClick={() => setIsolated((v) => !v)} aria-pressed={isolated}>
              Isolate
            </button>
          </div>
        </section>

        <aside className="panel">
          {!part && (
            <p className="panel-empty">
              Pick a part on the model, or search for one. {device.schematic
                ? 'Positions here are schematic, so no dimension lines are drawn — the numbers that are real are named and sourced on the part itself.'
                : 'Every number in this panel comes from the same Blender build script that exports the printable STLs, so the atlas cannot drift from the parts you actually print.'}
            </p>
          )}

          {part && (
            <>
              <div className="part-head">
                {part.code && <div className="part-code">{part.code}</div>}
                <h1 className="part-name">{part.name}</h1>
                <p className="part-role">{part.role}</p>
                <div className="part-tags">
                  {geo?.printed && <span className="tag" data-kind="printed">3D printed</span>}
                  {device.schematic && <span className="tag">Schematic</span>}
                  {part.flicker && (
                    <span className="tag" data-kind={part.flicker.role}>
                      {part.flicker.role === 'source' ? 'Flicker source'
                        : part.flicker.role === 'lever' ? 'Intervention point' : 'Light path'}
                    </span>
                  )}
                  {geo?.material && <span className="tag">{geo.material}</span>}
                </div>
              </div>

              <div className="block">
                <h2 className="block-head">What it does</h2>
                <p className="does">{part.does}</p>
              </div>

              {part.specs && (
                <div className="block">
                  <h2 className="block-head">Specification</h2>
                  <dl className="specs">
                    {part.specs.map(([k, v]) => (
                      <div key={k} style={{ display: 'contents' }}><dt>{k}</dt><dd>{v}</dd></div>
                    ))}
                  </dl>
                </div>
              )}

              {part.flicker && (
                <div className="block">
                  <h2 className="block-head">Photosensitivity</h2>
                  <p className="note" data-kind={part.flicker.role}>{part.flicker.note}</p>
                </div>
              )}

              {geo && !device.schematic && (
                <div className="block">
                  <h2 className="block-head">Measured from the model</h2>
                  <dl className="specs">
                    <div style={{ display: 'contents' }}>
                      <dt>Bounding box</dt><dd>{geo.sizeMm.map(round).join(' × ')} mm</dd>
                    </div>
                    {geo.printOrientation && (
                      <div style={{ display: 'contents' }}>
                        <dt>On the bed</dt><dd>{geo.printOrientation}</dd>
                      </div>
                    )}
                  </dl>
                </div>
              )}

              {part.interfaces && (
                <div className="block">
                  <h2 className="block-head">Bolted to</h2>
                  <ul className="list">{part.interfaces.map((i) => <li key={i}>{i}</li>)}</ul>
                </div>
              )}

              {part.watch && (
                <div className="block">
                  <h2 className="block-head">Watch out</h2>
                  <p className="note" data-kind="watch">{part.watch}</p>
                </div>
              )}

              {part.source && (
                <div className="block">
                  <h2 className="block-head">Where the number came from</h2>
                  <p className="note" data-kind="source">{part.source}</p>
                </div>
              )}
            </>
          )}
        </aside>
      </div>
    </>
  )
}
