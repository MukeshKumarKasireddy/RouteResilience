import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

/* ==========================================================================
   RouteResilience dashboard  (Person 2 – Visualization & Product)

   OPTIONAL REAL DATA: put a file at  public/data/network.json  like this:
   {
     "nodes": [ {"id": 0, "lat": 23.26, "lng": 77.41}, ... ],
     "edges": [ {"u": 0, "v": 1, "gap": false, "confidence": 0.9,
                 "length": 312.5, "criticality": 0.42}, ... ]
   }
   gap / confidence / length / criticality are optional.
   If the file is missing, a built-in demo network is shown.
   ========================================================================== */

const CONFIG = {
  center: [19.076, 72.8777], //mumbai
  zoom: 14,
  dataUrl: '/data/network.json',
}
const COLORS = { mask: '#2de2e6', rec: '#b6ff3b', net: '#ffffff', fail: '#ff2d55' }
const EPS = 1e-9

const STAGES = [
  { name: 'Imagery', layers: [], hint: 'Source satellite image of the study area.' },
  { name: 'Road mask', layers: ['mask'], hint: 'Roads the model found in the image. Gaps are where it lost the road.' },
  { name: 'Recovery', layers: ['mask', 'recovered'], hint: 'Green dashes are road sections the recovery step filled in. Hover one to see its confidence.' },
  { name: 'Network', layers: ['network'], hint: 'Recovered roads joined into one connected road network.' },
  { name: 'Critical roads', layers: ['network', 'critical'], hint: 'The roads most trips depend on. Brighter orange means more critical.' },
  { name: 'Failure', layers: ['network', 'critical'], after: true, hint: 'Pick a scenario on the left. Red roads are closed and red dots are cut off from the main network.' },
]

const LAYER_DEFS = [
  { key: 'mask', label: 'Roads found in the image', swatch: '#2de2e6' },
  { key: 'recovered', label: 'Recovered road gaps', swatch: 'repeating-linear-gradient(90deg,#b6ff3b 0 6px,transparent 6px 10px)' },
  { key: 'network', label: 'Road network', swatch: '#ffffff' },
  { key: 'critical', label: 'Critical roads', swatch: 'linear-gradient(90deg,#ffd640,#ff5a00)' },
]

/* ---------- helpers ---------- */
function mulberry32(a) {
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
function haversine(a, b) {
  const R = 6371000
  const rad = Math.PI / 180
  const dLat = (b.lat - a.lat) * rad
  const dLng = (b.lng - a.lng) * rad
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLng / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(s))
}
function heat(t) {
  const a = [255, 214, 64]
  const b = [255, 90, 0]
  return 'rgb(' + a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',') + ')'
}
const pct = (v) => Math.round(v * 100) + '%'

/* ---------- data ---------- */
function makeDemo() {
  const rand = mulberry32(11)
  const rows = 9
  const cols = 9
  const dLat = 0.0036
  const dLng = 0.0039
  const [c0, c1] = CONFIG.center
  const nodes = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      nodes.push({
        id: r * cols + c,
        lat: c0 + (r - (rows - 1) / 2) * dLat + (rand() - 0.5) * 0.0016,
        lng: c1 + (c - (cols - 1) / 2) * dLng + (rand() - 0.5) * 0.0016,
      })
    }
  }
  const cand = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const id = r * cols + c
      if (c < cols - 1) cand.push([id, id + 1])
      if (r < rows - 1) cand.push([id, id + cols])
      if (r < rows - 1 && c < cols - 1 && rand() < 0.12) cand.push([id, id + cols + 1])
    }
  }
  const parent = nodes.map((_, i) => i)
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]
      x = parent[x]
    }
    return x
  }
  const union = (a, b) => {
    parent[find(a)] = find(b)
  }
  const kept = []
  const dropped = []
  cand.forEach((p) => (rand() < 0.16 ? dropped : kept).push(p))
  kept.forEach(([u, v]) => union(u, v))
  dropped.forEach(([u, v]) => {
    if (find(u) !== find(v)) {
      union(u, v)
      kept.push([u, v])
    }
  })
  const edges = kept.map(([u, v]) => {
    const gap = rand() < 0.14
    return { u, v, gap, confidence: gap ? 0.55 + rand() * 0.4 : 1 }
  })
  return { nodes, edges }
}

function normalise(d) {
  const idx = new Map(d.nodes.map((n, i) => [n.id, i]))
  const nodes = d.nodes.map((n) => ({ lat: n.lat, lng: n.lng }))
  const edges = d.edges
    .map((e) => ({
      u: idx.get(e.u),
      v: idx.get(e.v),
      gap: !!e.gap,
      conf: e.confidence ?? 1,
      len: e.length,
      crit: e.criticality,
    }))
    .filter((e) => e.u !== undefined && e.v !== undefined)
  edges.forEach((e) => {
    if (!(e.len > 0)) e.len = haversine(nodes[e.u], nodes[e.v])
  })
  return { nodes, edges }
}

async function loadData() {
  try {
    const r = await fetch(CONFIG.dataUrl)
    if (!r.ok) throw new Error('HTTP ' + r.status)
    return { G: normalise(await r.json()), live: true }
  } catch (err) {
    console.info('No ' + CONFIG.dataUrl + ' – using demo network. (' + err.message + ')')
    return { G: normalise(makeDemo()), live: false }
  }
}

/* ---------- graph maths (demo-scale: fine up to a few hundred junctions) ---------- */
function dijkstra(adj, s, dead) {
  const n = adj.length
  const dist = new Array(n).fill(Infinity)
  const done = new Array(n).fill(false)
  dist[s] = 0
  for (;;) {
    let u = -1
    let best = Infinity
    for (let i = 0; i < n; i++) if (!done[i] && dist[i] < best) { best = dist[i]; u = i }
    if (u < 0) break
    done[u] = true
    for (const { v, e, w } of adj[u]) {
      if (dead && dead[e]) continue
      const nd = dist[u] + w
      if (nd < dist[v]) dist[v] = nd
    }
  }
  return dist
}

// Edge betweenness (Brandes): how many shortest routes use each road.
function edgeBetweenness(adj, m) {
  const n = adj.length
  const eb = new Array(m).fill(0)
  for (let s = 0; s < n; s++) {
    const dist = new Array(n).fill(Infinity)
    const sigma = new Array(n).fill(0)
    const pred = Array.from({ length: n }, () => [])
    const done = new Array(n).fill(false)
    const order = []
    dist[s] = 0
    sigma[s] = 1
    for (;;) {
      let u = -1
      let best = Infinity
      for (let i = 0; i < n; i++) if (!done[i] && dist[i] < best) { best = dist[i]; u = i }
      if (u < 0) break
      done[u] = true
      order.push(u)
      for (const { v, e, w } of adj[u]) {
        const nd = dist[u] + w
        if (nd < dist[v] - EPS) {
          dist[v] = nd
          sigma[v] = sigma[u]
          pred[v] = [[u, e]]
        } else if (Math.abs(nd - dist[v]) <= EPS) {
          sigma[v] += sigma[u]
          pred[v].push([u, e])
        }
      }
    }
    const delta = new Array(n).fill(0)
    while (order.length) {
      const w = order.pop()
      for (const [v, e] of pred[w]) {
        const c = (sigma[v] / sigma[w]) * (1 + delta[w])
        eb[e] += c
        delta[v] += c
      }
    }
  }
  const max = Math.max(...eb, EPS)
  return eb.map((x) => x / max)
}

function prepare(G) {
  const N = G.nodes.length
  const E = G.edges.length
  const adj = G.nodes.map(() => [])
  G.edges.forEach((e, i) => {
    adj[e.u].push({ v: e.v, e: i, w: e.len })
    adj[e.v].push({ v: e.u, e: i, w: e.len })
  })
  const crit = G.edges.every((e) => typeof e.crit === 'number') ? G.edges.map((e) => e.crit) : edgeBetweenness(adj, E)
  const order = [...crit.keys()].sort((a, b) => crit[b] - crit[a])
  const critCut = [...crit].sort((a, b) => b - a)[Math.max(0, Math.floor(E * 0.2))]
  const baseDist = []
  let sum = 0
  let pairs = 0
  for (let i = 0; i < N; i++) {
    baseDist[i] = dijkstra(adj, i, null)
    for (let j = i + 1; j < N; j++) {
      pairs++
      if (baseDist[i][j] < Infinity) sum += 1 / baseDist[i][j]
    }
  }
  const center = {
    lat: G.nodes.reduce((s, n) => s + n.lat, 0) / N,
    lng: G.nodes.reduce((s, n) => s + n.lng, 0) / N,
  }
  return { G, N, E, adj, crit, order, critCut, baseDist, baseEff: sum / pairs, center }
}

function failures(g, s) {
  const dead = new Array(g.E).fill(false)
  if (s.mode === 'targeted') {
    g.order.slice(0, s.n).forEach((i) => { dead[i] = true })
  } else if (s.mode === 'random') {
    const r = mulberry32(s.seed)
    const idx = [...Array(g.E).keys()]
    for (let i = g.E - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1))
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
    }
    idx.slice(0, Math.round((g.E * s.pct) / 100)).forEach((i) => { dead[i] = true })
  } else if (s.mode === 'zone' && s.zone) {
    g.G.edges.forEach((e, i) => {
      const a = g.G.nodes[e.u]
      const b = g.G.nodes[e.v]
      const mid = { lat: (a.lat + b.lat) / 2, lng: (a.lng + b.lng) / 2 }
      if (haversine(s.zone, mid) <= s.radius) dead[i] = true
    })
  } else if (s.mode === 'manual') {
    s.manual.forEach((i) => { dead[i] = true })
  }
  return dead
}

function measure(g, dead) {
  const { N, adj } = g
  const comp = new Array(N).fill(-1)
  const sizes = []
  for (let s = 0; s < N; s++) {
    if (comp[s] >= 0) continue
    const id = sizes.length
    let size = 0
    const q = [s]
    comp[s] = id
    while (q.length) {
      const u = q.pop()
      size++
      for (const { v, e } of adj[u]) {
        if (!dead[e] && comp[v] < 0) {
          comp[v] = id
          q.push(v)
        }
      }
    }
    sizes.push(size)
  }
  const main = sizes.indexOf(Math.max(...sizes))
  const cut = []
  comp.forEach((c, i) => { if (c !== main) cut.push(i) })
  let pairs = 0
  let conn = 0
  let sumB = 0
  let sumA = 0
  let eff = 0
  for (let i = 0; i < N; i++) {
    const d = dijkstra(adj, i, dead)
    for (let j = i + 1; j < N; j++) {
      pairs++
      if (d[j] < Infinity) {
        conn++
        sumA += d[j]
        sumB += g.baseDist[i][j]
        eff += 1 / d[j]
      }
    }
  }
  return {
    cut,
    tripsPossible: conn / pairs,
    longer: sumB > 0 ? sumA / sumB - 1 : 0,
    effLoss: Math.max(0, 1 - eff / pairs / g.baseEff),
  }
}

/* ---------- UI ---------- */
export default function App() {
  const [data, setData] = useState(null)
  useEffect(() => {
    let alive = true
    loadData().then((d) => { if (alive) setData(d) })
    return () => { alive = false }
  }, [])
  if (!data) return <div className="p-6 text-slate-700">Loading map…</div>
  return <Dashboard data={data} />
}

function Dashboard({ data }) {
  const { G, live } = data
  const g = useMemo(() => prepare(G), [G])

  const [mode, setMode] = useState('targeted')
  const [n, setN] = useState(5)
  const [pctFail, setPctFail] = useState(10)
  const [radius, setRadius] = useState(700)
  const [zone, setZone] = useState(null)
  const [seed, setSeed] = useState(7)
  const [manual, setManual] = useState([])
  const [after, setAfter] = useState(false)
  const [stage, setStage] = useState(0)
  const [basemap, setBasemap] = useState('sat')
  const [layers, setLayers] = useState({ mask: false, recovered: false, network: false, critical: false })

  const mapEl = useRef(null)
  const mapRef = useRef(null)
  const tilesRef = useRef(null)
  const groupsRef = useRef(null)

  const dead = useMemo(
    () => failures(g, { mode, n, pct: pctFail, seed, radius, zone, manual }),
    [g, mode, n, pctFail, seed, radius, zone, manual],
  )
  const m = useMemo(() => measure(g, dead), [g, dead])

  /* map: create once */
  useEffect(() => {
    const map = L.map(mapEl.current).setView(CONFIG.center, CONFIG.zoom)
    const tiles = {
      sat: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
        attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
      }),
      street: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors',
      }),
    }
    tiles.sat.addTo(map)
    map.fitBounds(L.latLngBounds(G.nodes.map((p) => [p.lat, p.lng])).pad(0.05))
    groupsRef.current = {
      mask: L.layerGroup().addTo(map),
      rec: L.layerGroup().addTo(map),
      net: L.layerGroup().addTo(map),
      crit: L.layerGroup().addTo(map),
      fail: L.layerGroup().addTo(map),
    }
    mapRef.current = map
    tilesRef.current = tiles
    const t = setTimeout(() => map.invalidateSize(), 100)
    return () => {
      clearTimeout(t)
      map.remove()
      mapRef.current = null
    }
  }, [G])

  /* map: background */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    Object.values(tilesRef.current).forEach((t) => map.removeLayer(t))
    tilesRef.current[basemap].addTo(map)
  }, [G, basemap])

  /* map: click to place disaster zone */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const onClick = (e) => {
      if (mode !== 'zone') return
      setZone({ lat: e.latlng.lat, lng: e.latlng.lng })
      setAfter(true)
    }
    map.on('click', onClick)
    return () => { map.off('click', onClick) }
  }, [G, mode])

  /* map: draw layers */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const gr = groupsRef.current
    Object.values(gr).forEach((x) => x.clearLayers())
    const isManual = mode === 'manual'
    const toggle = (i) => {
      setManual((prev) => (prev.includes(i) ? prev.filter((x) => x !== i) : [...prev, i]))
      setAfter(true)
    }
    const attach = (line, i) => {
      if (isManual) line.on('click', () => toggle(i))
      return line
    }

    G.edges.forEach((e, i) => {
      const a = G.nodes[e.u]
      const b = G.nodes[e.v]
      const pts = [[a.lat, a.lng], [b.lat, b.lng]]
      const hidden = after && dead[i]

      if (layers.mask && !e.gap) {
        L.polyline(pts, { color: COLORS.mask, weight: 3, opacity: 0.95, interactive: false }).addTo(gr.mask)
      }
      if (layers.recovered && e.gap) {
        L.polyline(pts, { color: COLORS.rec, weight: 4, dashArray: '6 6' })
          .bindTooltip('Recovered gap · ' + pct(e.conf) + ' confidence', { sticky: true })
          .addTo(gr.rec)
      }
      if (layers.network && !hidden) {
        L.polyline(pts, { color: '#0b1720', weight: 7, opacity: 0.55, interactive: false }).addTo(gr.net)
        attach(
          L.polyline(pts, { color: COLORS.net, weight: 3, opacity: 0.95 })
            .bindTooltip(isManual ? 'Click to close this road' : Math.round(e.len) + ' m', { sticky: true }),
          i,
        ).addTo(gr.net)
      }
      if (layers.critical && !hidden && g.crit[i] >= g.critCut) {
        attach(
          L.polyline(pts, { color: heat(g.crit[i]), weight: 4 + 5 * g.crit[i], opacity: 0.95 })
            .bindTooltip('Criticality ' + g.crit[i].toFixed(2) + ' · ' + Math.round(e.len) + ' m', { sticky: true }),
          i,
        ).addTo(gr.crit)
      }
      if (after && dead[i]) {
        attach(
          L.polyline(pts, { color: COLORS.fail, weight: 6, dashArray: '2 9', lineCap: 'round', opacity: 1 })
            .bindTooltip(isManual ? 'Closed – click to reopen' : 'Closed road', { sticky: true }),
          i,
        ).addTo(gr.fail)
      }
    })

    if (after) {
      m.cut.forEach((i) => {
        L.circleMarker([G.nodes[i].lat, G.nodes[i].lng], {
          radius: 6, color: '#fff', weight: 2, fillColor: COLORS.fail, fillOpacity: 1,
        })
          .bindTooltip('Cut off from the main network', { sticky: true })
          .addTo(gr.fail)
      })
      if (mode === 'zone' && zone) {
        L.circle([zone.lat, zone.lng], {
          radius, color: COLORS.fail, weight: 2, dashArray: '8 6',
          fillColor: COLORS.fail, fillOpacity: 0.12, interactive: false,
        }).addTo(gr.fail)
      }
    }
  }, [G, g, dead, m, layers, after, mode, zone, radius])

  /* handlers */
  const goStage = (i) => {
    setStage(i)
    setLayers({
      mask: STAGES[i].layers.includes('mask'),
      recovered: STAGES[i].layers.includes('recovered'),
      network: STAGES[i].layers.includes('network'),
      critical: STAGES[i].layers.includes('critical'),
    })
    setAfter(!!STAGES[i].after)
  }
  const changeMode = (next) => {
    setMode(next)
    if (next === 'zone' && !zone) setZone(g.center)
    if (next === 'manual') setLayers((l) => ({ ...l, network: true }))
    setAfter(true)
  }
  const showAfter = (v) => {
    setAfter(v)
    if (v) setLayers((l) => ({ ...l, network: true }))
  }
  const reset = () => {
    setManual([])
    setZone(mode === 'zone' ? g.center : null)
    setAfter(false)
  }

  const closed = dead.filter(Boolean).length
  const summary =
    closed === 0
      ? 'No roads are closed in this scenario.'
      : `${closed} ${closed === 1 ? 'road is' : 'roads are'} closed. ${m.cut.length} ${
          m.cut.length === 1 ? 'junction is' : 'junctions are'
        } cut off and trips are ${pct(m.longer)} longer on average.`
  const rows = [
    ['Trips still possible', pct(m.tripsPossible), m.tripsPossible],
    ['Junctions cut off', `${m.cut.length} of ${g.N}`, m.cut.length / g.N],
    ['Longer average trip', '+' + pct(m.longer), Math.min(1, m.longer)],
    ['Network efficiency lost', pct(m.effLoss), m.effLoss],
  ]

  const field = 'w-full rounded-md border border-slate-300 bg-white px-2.5 py-2 text-slate-900'
  const btn = 'mt-3 rounded-md border border-slate-900 bg-white px-3.5 py-1.5 text-sm font-semibold text-slate-900 hover:bg-slate-900 hover:text-white'

  return (
    <div className="grid min-h-screen grid-rows-[auto_minmax(0,1fr)] bg-slate-100 text-slate-900 lg:h-screen">
      <header className="flex flex-wrap items-center gap-x-8 gap-y-2 bg-slate-900 px-5 py-2.5 text-white">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-extrabold tracking-tight">RouteResilience</h1>
          <span className="rounded-full border border-white/40 px-2.5 py-0.5 text-xs text-slate-200">
            {live ? 'Live data' : 'Demo data'}
          </span>
        </div>
        <ol className="flex flex-wrap gap-1.5" aria-label="Pipeline stages">
          {STAGES.map((s, i) => (
            <li key={s.name}>
              <button
                type="button"
                onClick={() => goStage(i)}
                aria-current={i === stage ? 'step' : undefined}
                className={
                  'rounded-md border px-3 py-1.5 text-sm ' +
                  (i === stage
                    ? 'border-cyan-300 bg-cyan-300 text-slate-900'
                    : 'border-white/30 text-slate-200 hover:border-white')
                }
              >
                <b className="mr-1.5 font-semibold opacity-70">{i + 1}</b>
                {s.name}
              </button>
            </li>
          ))}
        </ol>
      </header>

      <main className="flex min-h-0 flex-col lg:grid lg:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="overflow-auto border-slate-300 lg:border-r">
          <section className="border-b border-slate-300 px-5 py-4">
            <h2 className="mb-3 text-lg font-bold">Scenario</h2>
            <label htmlFor="mode" className="mb-1 mt-2 block text-sm text-slate-600">What fails?</label>
            <select id="mode" className={field} value={mode} onChange={(e) => changeMode(e.target.value)}>
              <option value="targeted">The most critical roads</option>
              <option value="random">Random roads</option>
              <option value="zone">A disaster zone</option>
              <option value="manual">Roads I pick on the map</option>
            </select>

            {mode === 'targeted' && (
              <div>
                <label htmlFor="n" className="mb-1 mt-3 block text-sm text-slate-600">
                  Roads closed: <output className="font-semibold text-slate-900">{n}</output>
                </label>
                <input id="n" type="range" min="1" max="20" value={n} className="w-full accent-slate-900"
                  onChange={(e) => { setN(+e.target.value); setAfter(true) }} />
              </div>
            )}
            {mode === 'random' && (
              <div>
                <label htmlFor="pct" className="mb-1 mt-3 block text-sm text-slate-600">
                  Share of roads closed: <output className="font-semibold text-slate-900">{pctFail}%</output>
                </label>
                <input id="pct" type="range" min="1" max="40" value={pctFail} className="w-full accent-slate-900"
                  onChange={(e) => { setPctFail(+e.target.value); setAfter(true) }} />
                <button type="button" className={btn}
                  onClick={() => { setSeed(Math.floor(Math.random() * 1e6)); setAfter(true) }}>
                  Pick different roads
                </button>
              </div>
            )}
            {mode === 'zone' && (
              <div>
                <label htmlFor="radius" className="mb-1 mt-3 block text-sm text-slate-600">
                  Zone radius: <output className="font-semibold text-slate-900">{radius} m</output>
                </label>
                <input id="radius" type="range" min="200" max="2000" step="50" value={radius} className="w-full accent-slate-900"
                  onChange={(e) => { setRadius(+e.target.value); setAfter(true) }} />
                <p className="mt-2 text-sm text-slate-600">Click the map to move the zone.</p>
              </div>
            )}
            {mode === 'manual' && (
              <div>
                <p className="mt-3 text-sm text-slate-600">Click any road on the map to close it. Click it again to reopen.</p>
                <button type="button" className={btn} onClick={() => setManual([])}>Reopen all roads</button>
              </div>
            )}
            <div>
              <button type="button" className={btn} onClick={reset}>Reset scenario</button>
            </div>
          </section>

          <section className="border-b border-slate-300 px-5 py-4">
            <h2 className="mb-3 text-lg font-bold">Impact</h2>
            <p className="mb-3.5 font-medium">{summary}</p>
            {rows.map(([label, val, frac]) => (
              <div key={label} className="mb-3.5">
                <div className="flex justify-between gap-2 text-sm">
                  <span>{label}</span>
                  <span className="font-semibold tabular-nums">{val}</span>
                </div>
                <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-300">
                  <div className="h-full bg-slate-900 transition-all" style={{ width: Math.round(frac * 100) + '%' }} />
                </div>
              </div>
            ))}
          </section>

          <section className="px-5 py-4">
            <h2 className="mb-3 text-lg font-bold">Map layers</h2>
            {LAYER_DEFS.map((d) => (
              <label key={d.key} className="my-2 flex cursor-pointer items-center gap-2 text-slate-900">
                <input type="checkbox" checked={layers[d.key]}
                  onChange={(e) => setLayers((l) => ({ ...l, [d.key]: e.target.checked }))} />
                <span className="inline-block h-1.5 w-6 flex-none rounded-sm border border-black/25" style={{ background: d.swatch }} />
                {d.label}
              </label>
            ))}
            <label htmlFor="basemap" className="mb-1 mt-3 block text-sm text-slate-600">Background map</label>
            <select id="basemap" className={field} value={basemap} onChange={(e) => setBasemap(e.target.value)}>
              <option value="sat">Satellite</option>
              <option value="street">Street map</option>
            </select>
          </section>
        </aside>

        <div className="relative order-first h-[60vh] min-h-0 lg:order-none lg:h-auto">
          <div ref={mapEl} className="absolute inset-0 bg-slate-700" />
          <div className="absolute right-3 top-3 z-[1000] flex overflow-hidden rounded-lg bg-white shadow-md">
            {[['Before failure', false], ['After failure', true]].map(([label, v]) => (
              <button key={label} type="button" aria-pressed={after === v} onClick={() => showAfter(v)}
                className={'px-3.5 py-2 font-semibold ' + (after === v ? 'bg-slate-900 text-white' : 'bg-white text-slate-900')}>
                {label}
              </button>
            ))}
          </div>
          <div className="absolute bottom-6 left-3 z-[1000] max-w-[min(460px,calc(100%-24px))] rounded-lg bg-slate-900/90 px-3.5 py-2.5 text-sm text-white">
            {STAGES[stage].hint}
          </div>
        </div>
      </main>
    </div>
  )
}