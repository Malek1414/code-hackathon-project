import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

export interface GeometryRecord {
  id: string
  object: string
  node: string
  system: string
  explode: [number, number, number]
  explodeGl: [number, number, number]
  bboxMin: [number, number, number]
  bboxMax: [number, number, number]
  sizeMm: [number, number, number]
  centerM: [number, number, number]
  triangles: number
  printed: boolean
  printOrientation: string | null
  defaultHidden: boolean
  material: string | null
}

export interface ProjectedBox {
  /** screen-space corners of the selected part's bounding box, in CSS pixels */
  corners: { x: number; y: number }[]
  center: { x: number; y: number }
  /** true once the part is on screen and in front of the camera */
  visible: boolean
}

const INK = 0x14181b
const VERMILION = 0xd8341a

/** Bounding-box corner order used by the callout overlay. */
const CORNER_ORDER: [number, number, number][] = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
]

export class Viewer {
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls
  private raycaster = new THREE.Raycaster()
  private pointer = new THREE.Vector2()
  private host: HTMLElement
  private frame = 0
  private disposed = false

  /** id -> mesh */
  readonly meshes = new Map<string, THREE.Mesh>()
  private home = new Map<string, THREE.Vector3>()
  private explodeVec = new Map<string, THREE.Vector3>()
  private edges = new Map<string, THREE.LineSegments>()
  private baseColor = new Map<string, THREE.Color>()
  private systemOf = new Map<string, string>()

  private explodeAmount = 0
  private explodeTarget = 0
  private selected: string | null = null
  private hovered: string | null = null
  private isolated = false
  private visibleIds = new Set<string>()
  /** per-part colour override, used by the flicker-chain layer */
  private accent: Record<string, string> | null = null
  /** parts drawn as translucent enclosures rather than solids */
  private ghosted = new Set<string>()

  onPick: (id: string | null) => void = () => {}
  onHover: (id: string | null) => void = () => {}
  onProject: (box: ProjectedBox | null) => void = () => {}

  constructor(host: HTMLElement) {
    this.host = host
    const { clientWidth: w, clientHeight: h } = host

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    this.renderer.setSize(w, h, false)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    host.appendChild(this.renderer.domElement)

    this.camera = new THREE.PerspectiveCamera(38, w / h, 0.01, 60)
    this.camera.position.set(0.34, 0.30, 0.42)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.target.set(0, 0.06, 0)
    this.controls.minDistance = 0.12
    this.controls.maxDistance = 1.6
    this.controls.enablePan = false

    // Flat, diagrammatic lighting: the atlas is a drawing, not a product render.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa6ad, 2.1))
    const key = new THREE.DirectionalLight(0xffffff, 1.5)
    key.position.set(-0.6, 1.0, 0.8)
    this.scene.add(key)
    const fill = new THREE.DirectionalLight(0xcfd8de, 0.7)
    fill.position.set(0.8, 0.2, -0.3)
    this.scene.add(fill)

    this.renderer.domElement.addEventListener('pointermove', this.handleMove)
    this.renderer.domElement.addEventListener('pointerdown', this.handleDown)
    this.renderer.domElement.addEventListener('pointerup', this.handleUp)
  }

  // ---------------------------------------------------------------- loading
  async load(
    url: string,
    records: GeometryRecord[],
    colorFor: (system: string) => string,
    ghostFor: (system: string) => boolean = () => false,
  ) {
    const gltf = await new GLTFLoader().loadAsync(url)
    const byNode = new Map(records.map((r) => [r.node, r]))

    gltf.scene.updateMatrixWorld(true)
    const collected: THREE.Mesh[] = []
    gltf.scene.traverse((o) => { if ((o as THREE.Mesh).isMesh) collected.push(o as THREE.Mesh) })

    for (const mesh of collected) {
      const rec = byNode.get(mesh.name)
      if (!rec) continue

      // Flatten the glTF hierarchy: parts move independently when exploded, so a
      // parented transform would drag children along with them.
      mesh.updateMatrixWorld(true)
      const world = mesh.matrixWorld.clone()
      mesh.geometry = mesh.geometry.clone()
      mesh.geometry.applyMatrix4(world)
      mesh.position.set(0, 0, 0)
      mesh.rotation.set(0, 0, 0)
      mesh.scale.set(1, 1, 1)
      mesh.matrix.identity()
      mesh.matrixWorld.identity()

      const color = new THREE.Color(colorFor(rec.system))
      mesh.material = new THREE.MeshStandardMaterial({
        color, roughness: 0.72, metalness: 0.04,
        transparent: true, opacity: 1,
        depthWrite: !ghostFor(rec.system),
      })
      mesh.userData.id = rec.id
      mesh.renderOrder = 1

      this.scene.add(mesh)
      this.meshes.set(rec.id, mesh)
      this.home.set(rec.id, mesh.position.clone())
      this.explodeVec.set(rec.id, new THREE.Vector3(...rec.explodeGl))
      this.baseColor.set(rec.id, color)
      this.systemOf.set(rec.id, rec.system)
      if (ghostFor(rec.system)) this.ghosted.add(rec.id)
      this.visibleIds.add(rec.id)

      // Hairline edges are what make this read as a drawing rather than a render.
      const eg = new THREE.EdgesGeometry(mesh.geometry, 28)
      const line = new THREE.LineSegments(
        eg,
        new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.22 }),
      )
      line.renderOrder = 2
      this.scene.add(line)
      this.edges.set(rec.id, line)
    }

    // A silently skipped mesh is the failure mode of this loader: a name that does
    // not match simply never appears. Say so rather than rendering a partial model.
    const missing = records.filter((r) => !this.meshes.has(r.id)).map((r) => r.node)
    if (missing.length) console.warn('[atlas] unmatched nodes:', missing)
    console.info(`[atlas] ${this.meshes.size}/${records.length} parts loaded`)

    this.reframe()
    this.tick()
  }

  /**
   * Frame the visible parts at the explode amount they are heading for, not the
   * one they are at, so switching preset does not leave the camera chasing parts
   * that have already left the frame.
   *
   * The tripod legs are nearly a metre of geometry around a 112 mm pod, so the
   * subject box deliberately excludes context and ghosted enclosures: frame the
   * product and let the rest run off the sheet.
   */
  reframe() {
    const all = new THREE.Box3()
    const subject = new THREE.Box3()
    const at = new THREE.Vector3()
    const b = new THREE.Box3()

    for (const [id, m] of this.meshes) {
      if (!this.visibleIds.has(id)) continue
      if (this.isolated && this.selected !== id) continue
      m.geometry.computeBoundingBox()
      at.copy(this.home.get(id)!).addScaledVector(this.explodeVec.get(id)!, this.explodeTarget)
      b.copy(m.geometry.boundingBox!).translate(at)
      all.union(b)
      if (this.systemOf.get(id) !== 'external' && !this.ghosted.has(id)) subject.union(b)
    }

    const box = subject.isEmpty() ? all : subject
    if (box.isEmpty()) return

    const c = box.getCenter(new THREE.Vector3())
    this.controls.target.copy(c)
    const r = box.getBoundingSphere(new THREE.Sphere()).radius
    const dist = (r / Math.sin((this.camera.fov * Math.PI) / 360)) * 1.58
    const dir = new THREE.Vector3(0.58, 0.42, 0.74).normalize()
    this.camera.position.copy(c).addScaledVector(dir, dist)
    this.controls.maxDistance = Math.max(1.6, dist * 2.4)
    this.controls.update()
  }

  // ----------------------------------------------------------------- state
  setVisible(ids: Set<string>) {
    this.visibleIds = ids
    for (const [id, m] of this.meshes) {
      const on = ids.has(id) && (!this.isolated || this.selected === id)
      m.visible = on
      const e = this.edges.get(id)
      if (e) e.visible = on
    }
  }

  setExplode(t: number) { this.explodeTarget = t }

  setSelected(id: string | null) {
    this.selected = id
    if (this.isolated) this.setVisible(this.visibleIds)
    this.paint()
  }

  setHovered(id: string | null) { this.hovered = id; this.paint() }

  setAccent(accent: Record<string, string> | null) { this.accent = accent; this.paint() }

  setIsolated(on: boolean) { this.isolated = on; this.setVisible(this.visibleIds) }

  focusSelected() { this.reframe() }

  private paint() {
    for (const [id, m] of this.meshes) {
      const mat = m.material as THREE.MeshStandardMaterial
      const override = this.accent?.[id]
      const base = override ? new THREE.Color(override) : this.baseColor.get(id)!
      const edge = this.edges.get(id)?.material as THREE.LineBasicMaterial | undefined
      if (this.selected === id) {
        mat.color.set(VERMILION); mat.opacity = 1
        if (edge) { edge.color.set(VERMILION); edge.opacity = 0.85 }
      } else if (this.hovered === id) {
        mat.color.copy(base).lerp(new THREE.Color(VERMILION), 0.32); mat.opacity = 1
        if (edge) { edge.color.set(INK); edge.opacity = 0.45 }
      } else {
        mat.color.copy(base)
        const ghost = this.ghosted.has(id) && !this.isolated
        if (ghost) mat.opacity = this.selected ? 0.1 : 0.2
        else mat.opacity = this.selected && !this.isolated ? 0.44 : 1
        if (edge) {
          edge.color.set(INK)
          edge.opacity = ghost ? 0.18 : this.selected ? 0.15 : 0.22
        }
      }
    }
  }

  // -------------------------------------------------------------- pointer
  private downAt = { x: 0, y: 0 }
  private handleDown = (e: PointerEvent) => { this.downAt = { x: e.clientX, y: e.clientY } }

  private handleUp = (e: PointerEvent) => {
    // Distinguish a tap from the end of an orbit drag, or every rotation
    // would also change the selection.
    const moved = Math.hypot(e.clientX - this.downAt.x, e.clientY - this.downAt.y)
    if (moved > 5) return
    this.onPick(this.hit(e))
  }

  private handleMove = (e: PointerEvent) => {
    const id = this.hit(e)
    if (id !== this.hovered) { this.setHovered(id); this.onHover(id) }
    this.renderer.domElement.style.cursor = id ? 'pointer' : 'grab'
  }

  private hit(e: PointerEvent): string | null {
    const r = this.renderer.domElement.getBoundingClientRect()
    this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const targets = [...this.meshes.values()].filter((m) => m.visible)
    const hits = this.raycaster.intersectObjects(targets, false)
    return hits.length ? (hits[0].object.userData.id as string) : null
  }

  // ----------------------------------------------------------------- loop
  private tick = () => {
    if (this.disposed) return
    this.frame = requestAnimationFrame(this.tick)

    this.explodeAmount += (this.explodeTarget - this.explodeAmount) * 0.14
    for (const [id, m] of this.meshes) {
      const home = this.home.get(id)!
      const ex = this.explodeVec.get(id)!
      m.position.copy(home).addScaledVector(ex, this.explodeAmount)
      const e = this.edges.get(id)
      if (e) e.position.copy(m.position)
    }

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
    this.project()
  }

  private project() {
    const id = this.selected
    const mesh = id ? this.meshes.get(id) : null
    if (!id || !mesh || !mesh.visible) { this.onProject(null); return }

    mesh.geometry.computeBoundingBox()
    const bb = mesh.geometry.boundingBox!
    const r = this.renderer.domElement.getBoundingClientRect()
    const v = new THREE.Vector3()
    const corners: { x: number; y: number }[] = []
    let anyInFront = false

    for (const [fx, fy, fz] of CORNER_ORDER) {
      v.set(fx ? bb.max.x : bb.min.x, fy ? bb.max.y : bb.min.y, fz ? bb.max.z : bb.min.z)
      v.add(mesh.position)
      v.project(this.camera)
      if (v.z < 1) anyInFront = true
      corners.push({ x: ((v.x + 1) / 2) * r.width, y: ((-v.y + 1) / 2) * r.height })
    }
    const center = corners.reduce((a, c) => ({ x: a.x + c.x / 8, y: a.y + c.y / 8 }), { x: 0, y: 0 })
    this.onProject({ corners, center, visible: anyInFront })
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.host
    if (!w || !h) return
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h, false)
  }

  dispose() {
    this.disposed = true
    cancelAnimationFrame(this.frame)
    this.renderer.domElement.removeEventListener('pointermove', this.handleMove)
    this.renderer.domElement.removeEventListener('pointerdown', this.handleDown)
    this.renderer.domElement.removeEventListener('pointerup', this.handleUp)
    this.controls.dispose()
    this.renderer.dispose()
    this.renderer.domElement.remove()
  }
}
