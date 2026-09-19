import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

/* ==========================================================================
   RouteResilience dashboard  (Person 2 – Visualization & Product)

   WHO PROVIDES WHAT (from the team task document)
     Mukesh    -> satellite imagery + the reconstructed road network (nodes, edges)
     Suryakant -> detected roads and recovered gaps ("gap": true + "confidence")
     Manogya   -> criticality + failure scenarios + before/after metrics ("scenarios")
     Paridhi   -> this dashboard: shows all of the above

   AREAS: one entry per study area. To add an area, copy the Mumbai line,
   change id / name / center / dataUrl, and put its file in public/data/.
   If an area has no data file yet, a demo network is drawn instead.

   DATA FILE (what Mukesh and Manogya send – public/data/network.json):
   {
     "nodes": [ {"id": 0, "lat": 19.07, "lng": 72.87}, ... ],
     "edges": [ {"u": 0, "v": 1, "gap": false, "confidence": 0.9,
                 "length": 312.5, "criticality": 0.42}, ... ],
     "scenarios": [                       <- optional, from the network analysis
       { "id": "critical-1",
         "name": "Most critical road fails",
         "failed_edges": [12],            <- positions in "edges" (first edge = 0)
         "zone": {"lat": 19.07, "lng": 72.87, "radius": 700},   <- optional
         "metrics": { "trips_possible": 0.74, "isolated_nodes": 12,
                      "path_length_increase": 0.04, "efficiency_loss": 0.16 },
         "extra_metrics": [ {"label": "Reachable area", "before": "12 km2", "after": "9 km2"} ]
       }
     ]
   }
   Everything except nodes/edges u,v is optional.
   ========================================================================== */

const AREAS = [
  { id: 'mumbai', name: 'Mumbai', center: [19.076, 72.8777], zoom: 14, dataUrl: '/data/network.json' },
  // { id: 'pune', name: 'Pune', center: [18.5204, 73.8567], zoom: 14, dataUrl: '/data/pune/network.json' },
]

const COLORS = { mask: '#2de2e6', rec: '#b6ff3b', net: '#ffffff', fail: '#ff2d55' }
const EPS = 1e-9

const STAGES = [
  { name: 'Imagery', from: 'Mukesh (geospatial data)', layers: [], hint: 'Source satellite image of the study area.' },
  { name: 'Detected roads', from: 'Suryakant (road segmentation)', layers: ['detected'], hint: 'Roads the model found in the image. Gaps are where it lost the road.' },
  { name: 'Recovery', from: 'Suryakant (gap recovery)', layers: ['detected', 'recovered'], hint: 'Green dashes are road sections the recovery step filled in. Hover one to see its confidence.' },
  { name: 'Network', from: 'Mukesh (road network reconstruction)', layers: ['reconstructed'], hint: 'Recovered roads joined into one connected road network.' },
  { name: 'Critical roads', from: 'Manogya (criticality analysis)', layers: ['reconstructed', 'critical'], hint: 'The roads most trips depend on. Brighter orange means more critical.' },
  { name: 'Failure', from: 'Manogya (failure simulation)', layers: ['reconstructed', 'critical'], after: true, hint: 'Pick a scenario on the left. Red roads are closed and red dots are cut off from the main network.' },
]

const LAYER_DEFS = [
  { key: 'satellite', label: 'Satellite imagery', swatch: '#5b7a4a' },
  { key: 'detected', label: 'Detected roads', swatch: '#2de2e6' },
  { key: 'recovered', label: 'Recovered road gaps', swatch: 'repeating-linear-gradient(90deg,#b6ff3b 0 6px,transparent 6px 10px)' },
  { key: 'reconstructed', label: 'Reconstructed roads', swatch: '#ffffff' },
  { key: 'critical', label: 'Critical roads', swatch: 'linear-gradient(90deg,#ffd640,#ff5a00)' },
  { key: 'failed', label: 'Failed roads', swatch: 'repeating-linear-gradient(90deg,#ff2d55 0 3px,transparent 3px 8px)' },
  { key: 'affected', label: 'Affected areas', swatch: 'rgba(255,45,85,.5)' },
]

const BASE_MODES = [
  { v: 'single', label: 'One road (click it on the map)' },
  { v: 'targeted', label: 'The most critical roads' },
  { v: 'zone', label: 'A geographic disaster zone' },
  { v: 'manual', label: 'Several roads (click them on the map)' },
  { v: 'random', label: 'Random roads' },
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
function makeDemo(center) {
  const rand = mulberry32(11)
  const rows = 9
  const cols = 9
  const dLat = 0.0036
  const dLng = 0.0039
  const [c0, c1] = center
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
  const edges = []
  const srcToIdx = new Map()
  d.edges.forEach((e, k) => {
    const u = idx.get(e.u)
    const v = idx.get(e.v)
    if (u === undefined || v === undefined) return
    srcToIdx.set(k, edges.length)
    edges.push({ u, v, gap: !!e.gap, conf: e.confidence ?? 1, len: e.length, crit: e.criticality })
  })
  edges.forEach((e) => {
    if (!(e.len > 0)) e.len = haversine(nodes[e.u], nodes[e.v])
  })
  const scenarios = (d.scenarios || []).map((s, k) => ({
    id: String(s.id ?? k),
    name: s.name ?? 'Scenario ' + (k + 1),
    failed: (s.failed_edges || []).map((x) => srcToIdx.get(x)).filter((x) => x !== undefined),
    zone: s.zone || null,
    metrics: s.metrics || {},
    extra: s.extra_metrics || [],
  }))
  return { nodes, edges, scenarios }
}

async function loadData(area) {
  try {
    const r = await fetch(area.dataUrl)
    if (!r.ok) throw new Error('HTTP ' + r.status)
    return { G: normalise(await r.json()), live: true }
  } catch (err) {
    console.info('No ' + area.dataUrl + ' – using demo network. (' + err.message + ')')
    return { G: normalise(makeDemo(area.center)), live: false }
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
  const gaps = G.edges.filter((e) => e.gap)
  const avgConf = gaps.length ? gaps.reduce((s, e) => s + e.conf, 0) / gaps.length : null
  return { G, N, E, adj, crit, order, critCut, baseDist, baseEff: sum / pairs, center, gapCount: gaps.length, avgConf }
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
  } else if (s.mode === 'manual' || s.mode === 'single') {
    s.manual.forEach((i) => { dead[i] = true })
  } else if (s.mode === 'analysis' && s.scenario) {
    s.scenario.failed.forEach((i) => { dead[i] = true })
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
  const [areaId, setAreaId] = useState(AREAS[0].id)
  const area = AREAS.find((a) => a.id === areaId)
  const [data, setData] = useState(null)
  useEffect(() => {
    let alive = true
    setData(null)
    loadData(area).then((d) => { if (alive) setData(d) })
    return () => { alive = false }
  }, [area])
  if (!data) return <div className="p-6 text-slate-700">Loading {area.name}…</div>
  return <Dashboard key={area.id} data={data} area={area} onArea={setAreaId} />
}

function Dashboard({ data, area, onArea }) {
  const { G, live } = data
  const g = useMemo(() => prepare(G), [G])
  const hasScenarios = G.scenarios.length > 0
  const modes = hasScenarios ? [{ v: 'analysis', label: 'Scenario from the network analysis' }, ...BASE_MODES] : BASE_MODES

  const [mode, setMode] = useState(hasScenarios ? 'analysis' : 'targeted')
  const [scenarioId, setScenarioId] = useState(hasScenarios ? G.scenarios[0].id : null)
  const [n, setN] = useState(3)
  const [pctFail, setPctFail] = useState(10)
  const [radius, setRadius] = useState(700)
  const [zone, setZone] = useState(null)
  const [seed, setSeed] = useState(7)
  const [manual, setManual] = useState([])
  const [after, setAfter] = useState(false)
  const [stage, setStage] = useState(0)
  const [layers, setLayers] = useState({
    satellite: true, detected: false, recovered: false, reconstructed: false, critical: false, failed: true, affected: true,
  })

  const mapEl = useRef(null)
  const mapRef = useRef(null)
  const tilesRef = useRef(null)
  const groupsRef = useRef(null)

  const scenario = mode === 'analysis' ? G.scenarios.find((s) => s.id === scenarioId) || null : null
  const dead = useMemo(
    () => failures(g, { mode, n, pct: pctFail, seed, radius, zone, manual, scenario }),
    [g, mode, n, pctFail, seed, radius, zone, manual, scenario],
  )
  const m = useMemo(() => measure(g, dead), [g, dead])
  const sm = scenario ? scenario.metrics : {}
  const shown = {
    cutCount: sm.isolated_nodes ?? m.cut.length,
    trips: sm.trips_possible ?? m.tripsPossible,
    longer: sm.path_length_increase ?? m.longer,
    effLoss: sm.efficiency_loss ?? m.effLoss,
  }
  const zoneShown =
    mode === 'zone' && zone ? { lat: zone.lat, lng: zone.lng, radius }
    : mode === 'analysis' && scenario && scenario.zone ? scenario.zone
    : null

  /* map: create once */
  useEffect(() => {
    const map = L.map(mapEl.current).setView(area.center, area.zoom)
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
  }, [G, area])

  /* map: satellite imagery on / off (off shows a street map) */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    Object.values(tilesRef.current).forEach((t) => map.removeLayer(t))
    tilesRef.current[layers.satellite ? 'sat' : 'street'].addTo(map)
  }, [G, area, layers.satellite])

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
  }, [G, area, mode])

  /* map: draw layers */
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const gr = groupsRef.current
    Object.values(gr).forEach((x) => x.clearLayers())
    const clickable = mode === 'manual' || mode === 'single'
    const pick = (i) => {
      setManual((prev) => {
        if (mode === 'single') return prev.length === 1 && prev[0] === i ? [] : [i]
        return prev.includes(i) ? prev.filter((x) => x !== i) : [...prev, i]
      })
      setAfter(true)
    }
    const attach = (line, i) => {
      if (clickable) line.on('click', () => pick(i))
      return line
    }

    G.edges.forEach((e, i) => {
      const a = G.nodes[e.u]
      const b = G.nodes[e.v]
      const pts = [[a.lat, a.lng], [b.lat, b.lng]]
      const hidden = after && dead[i]

      if (layers.detected && !e.gap) {
        L.polyline(pts, { color: COLORS.mask, weight: 3, opacity: 0.95, interactive: false }).addTo(gr.mask)
      }
      if (layers.recovered && e.gap) {
        L.polyline(pts, { color: COLORS.rec, weight: 4, dashArray: '6 6' })
          .bindTooltip('Recovered gap · ' + pct(e.conf) + ' confidence', { sticky: true })
          .addTo(gr.rec)
      }
      if (layers.reconstructed && !hidden) {
        L.polyline(pts, { color: '#0b1720', weight: 7, opacity: 0.55, interactive: false }).addTo(gr.net)
        attach(
          L.polyline(pts, { color: COLORS.net, weight: 3, opacity: 0.95 })
            .bindTooltip(clickable ? 'Click to close this road' : Math.round(e.len) + ' m', { sticky: true }),
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
      if (after && layers.failed && dead[i]) {
        attach(
          L.polyline(pts, { color: COLORS.fail, weight: 6, dashArray: '2 9', lineCap: 'round', opacity: 1 })
            .bindTooltip(clickable ? 'Closed – click to reopen' : 'Closed road', { sticky: true }),
          i,
        ).addTo(gr.fail)
      }
    })

    if (after && layers.affected) {
      m.cut.forEach((i) => {
        L.circleMarker([G.nodes[i].lat, G.nodes[i].lng], {
          radius: 6, color: '#fff', weight: 2, fillColor: COLORS.fail, fillOpacity: 1,
        })
          .bindTooltip('Cut off from the main network', { sticky: true })
          .addTo(gr.fail)
      })
      if (zoneShown) {
        L.circle([zoneShown.lat, zoneShown.lng], {
          radius: zoneShown.radius, color: COLORS.fail, weight: 2, dashArray: '8 6',
          fillColor: COLORS.fail, fillOpacity: 0.12, interactive: false,
        }).addTo(gr.fail)
      }
    }
  }, [G, g, dead, m, layers, after, mode, zoneShown && zoneShown.lat, zoneShown && zoneShown.lng, zoneShown && zoneShown.radius])

  /* handlers */
  const goStage = (i) => {
    const on = STAGES[i].layers
    setStage(i)
    setLayers((l) => ({
      ...l,
      detected: on.includes('detected'),
      recovered: on.includes('recovered'),
      reconstructed: on.includes('reconstructed'),
      critical: on.includes('critical'),
      failed: true,
      affected: true,
    }))
    setAfter(!!STAGES[i].after)
  }
  const changeMode = (next) => {
    setMode(next)
    if (next === 'zone' && !zone) setZone(g.center)
    if (next === 'single') setManual((p) => p.slice(0, 1))
    if (next === 'manual' || next === 'single') setLayers((l) => ({ ...l, reconstructed: true, failed: true }))
    setAfter(true)
  }
  const showAfter = (v) => {
    setAfter(v)
    if (v) setLayers((l) => ({ ...l, reconstructed: true }))
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
      : `${closed} ${closed === 1 ? 'road is' : 'roads are'} closed. ${shown.cutCount} ${
          shown.cutCount === 1 ? 'junction is' : 'junctions are'
        } cut off and trips are ${pct(shown.longer)} longer on average.`
  const connectedAfter = Math.max(0, g.N - shown.cutCount)
  const rows = [
    { label: 'Trips still possible', before: '100%', after: pct(shown.trips), bf: 1, af: shown.trips },
    { label: 'Junctions in the main network', before: String(g.N), after: String(connectedAfter), bf: 1, af: connectedAfter / g.N },
    { label: 'Average trip length', before: 'baseline', after: '+' + pct(shown.longer), bf: 1 / (1 + shown.longer), af: 1 },
    { label: 'Network efficiency', before: '100%', after: pct(1 - shown.effLoss), bf: 1, af: Math.max(0, 1 - shown.effLoss) },
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
          {AREAS.length > 1 && (
            <select aria-label="Study area" value={area.id} onChange={(e) => onArea(e.target.value)}
              className="rounded-md border border-white/40 bg-slate-800 px-2 py-1 text-sm text-white">
              {AREAS.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select>
          )}
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
              {modes.map((o) => <option key={o.v} value={o.v}>{o.label}</option>)}
            </select>

            {mode === 'analysis' && (
              <div>
                <label htmlFor="scenario" className="mb-1 mt-3 block text-sm text-slate-600">Scenario</label>
                <select id="scenario" className={field} value={scenarioId}
                  onChange={(e) => { setScenarioId(e.target.value); setAfter(true) }}>
                  {G.scenarios.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
                </select>
              </div>
            )}
            {mode === 'single' && (
              <p className="mt-3 text-sm text-slate-600">Click one road on the map to close it. Click another to switch, or the same one to reopen.</p>
            )}
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
            <p className="mb-3 font-medium">{summary}</p>
            <div className="mb-3 flex gap-4 text-xs text-slate-600">
              <span className="flex items-center gap-1.5"><i className="inline-block h-1.5 w-4 rounded-full bg-slate-400" />Before failure</span>
              <span className="flex items-center gap-1.5"><i className="inline-block h-1.5 w-4 rounded-full bg-slate-900" />After failure</span>
            </div>
            {rows.map((r) => (
              <div key={r.label} className="mb-3.5">
                <div className="flex justify-between gap-2 text-sm">
                  <span>{r.label}</span>
                  <span className="font-semibold tabular-nums">{r.before} → {r.after}</span>
                </div>
                <div className="mt-1.5 space-y-1">
                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-200">
                    <div className="h-full bg-slate-400 transition-all" style={{ width: Math.round(r.bf * 100) + '%' }} />
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-slate-200">
                    <div className="h-full bg-slate-900 transition-all" style={{ width: Math.round(r.af * 100) + '%' }} />
                  </div>
                </div>
              </div>
            ))}
            {scenario && scenario.extra.map((x) => (
              <div key={x.label} className="mb-2 flex justify-between gap-2 text-sm">
                <span>{x.label}</span>
                <span className="font-semibold tabular-nums">{String(x.before)} → {String(x.after)}</span>
              </div>
            ))}
            {g.avgConf !== null && (
              <p className="mt-3 text-sm text-slate-600">
                Recovery: {g.gapCount} road sections filled in, average confidence {pct(g.avgConf)}.
              </p>
            )}
            <p className="mt-2 text-xs text-slate-500">
              {scenario && Object.keys(scenario.metrics).length > 0
                ? 'Numbers come from the network analysis.'
                : 'Numbers are calculated in this browser from the current network.'}
            </p>
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
            <p className="mt-2 text-xs text-slate-500">Failed roads and affected areas appear in the After failure view. With satellite imagery off, a street map is shown.</p>
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
            <div className="mt-1 text-xs text-slate-300">
              {live ? 'Data from' : 'Demo data – real data will come from'} {STAGES[stage].from}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}