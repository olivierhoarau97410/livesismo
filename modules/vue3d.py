"""
vue3d.py — Vue 3D Three.js : terrain Réunion + séismes en profondeur
Inspiré de cartoRunv4.html (Desktop/mes-applis/Carto RUN/)
Rendu entièrement côté navigateur → la caméra ne se réinitialise jamais.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

DATA_DIR  = Path(__file__).parent.parent / "data"
TERRAIN_F = DATA_DIR / "terrain_250m.npz"

PF_LAT, PF_LON   = -21.244, 55.712
PDN_LAT, PDN_LON = -21.091, 55.483

# ── Couleur séisme par magnitude ──────────────────────────────────────────────
def _mag_color(mag):
    if pd.isna(mag) or mag < 1: return "#a0a0a0"
    if mag < 2:  return "#4fc3f7"
    if mag < 3:  return "#81c784"
    if mag < 4:  return "#fb8c00"
    if mag < 5:  return "#ef5350"
    return "#b71c1c"

def _mag_size(mag):
    if pd.isna(mag): return 0.3
    return max(0.2, min(1.5, 0.15 * (mag + 1.5) ** 1.5))


# ── Chargement / génération du terrain ───────────────────────────────────────

def _load_terrain(max_pts=120):
    if TERRAIN_F.exists():
        npz = np.load(TERRAIN_F)
        lat, lon, Z = npz["lat"], npz["lon"], npz["Z"]
        # Downsample pour le rendu temps réel
        step_r = max(1, len(lat) // max_pts)
        step_c = max(1, len(lon) // max_pts)
        return lat[::step_r], lon[::step_c], Z[::step_r, ::step_c], False
    # Terrain synthétique
    lat = np.linspace(-21.84, -20.88, max_pts)
    lon = np.linspace(55.13,  55.92, max_pts)
    LON, LAT = np.meshgrid(lon, lat)
    Z  = 3070 * np.exp(-((LAT-PDN_LAT)**2/0.04 + (LON-PDN_LON)**2/0.04))
    Z += 2632 * np.exp(-((LAT-PF_LAT )**2/0.02 + (LON-PF_LON )**2/0.02))
    Z  = gaussian_filter(Z, sigma=3)
    return lat, lon, Z, True


# ── Construction de l'HTML Three.js ──────────────────────────────────────────

def make_vue3d_html(df: pd.DataFrame, vert_exag: float = 4.0) -> str:
    lat_1d, lon_1d, Z, synthetic = _load_terrain()

    nrows, ncols = Z.shape
    z_min = float(np.nanmin(Z))
    z_max = float(np.nanmax(Z))

    # Normalisation terrain → cube [-30, 30]
    SCALE = 60.0
    lat_range = float(lat_1d[-1] - lat_1d[0])
    lon_range = float(lon_1d[-1] - lon_1d[0])

    def norm_lon(v): return (v - float(lon_1d[0])) / lon_range * SCALE - SCALE/2
    def norm_lat(v): return (v - float(lat_1d[0])) / lat_range * SCALE - SCALE/2
    def norm_z(v):   return (v - z_min) / max(z_max - z_min, 1) * SCALE * 0.5 * (vert_exag/4)

    # Grille terrain → JSON compact (liste de hauteurs normalisées)
    Z_norm = np.nan_to_num(Z, nan=0.0)
    heights = [[round(norm_z(float(Z_norm[r, c])), 3) for c in range(ncols)] for r in range(nrows)]
    heights_json = json.dumps(heights)

    # Séismes → JSON
    # Échelle visuelle profondeur : 40 km → -40 unités (même ordre que la hauteur du terrain ~30u)
    DEPTH_SCALE = 1.0   # 1 km = 1 unité Three.js

    quakes = []
    if not df.empty:
        for _, row in df.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            depth = row.get("profondeur_km", 0)
            mag   = row.get("magnitude", np.nan)
            date  = str(row.get("date", ""))[:16]
            if pd.isna(lat) or pd.isna(lon): continue
            depth = 0 if pd.isna(depth) else float(depth)
            # depth en km → unités Three.js (négatif = sous la surface)
            y_pos = -depth * DEPTH_SCALE
            quakes.append({
                "x": round(norm_lon(float(lon)), 3),
                "z": round(norm_lat(float(lat)), 3),
                "y": round(y_pos, 3),
                "r": round(_mag_size(mag), 3),
                "c": _mag_color(mag),
                "label": f"M{mag:.1f} | {depth:.1f} km | {date} UTC" if not pd.isna(mag) else f"{depth:.1f} km | {date}",
            })
    quakes_json = json.dumps(quakes)

    # Marqueurs sommets
    markers = [
        {"x": round(norm_lon(PF_LON),  3), "z": round(norm_lat(PF_LAT),  3),
         "y": round(norm_z(2632), 3) + 1.5, "label": "🌋 Piton de la Fournaise  2632 m", "c": "#ef5350"},
        {"x": round(norm_lon(PDN_LON), 3), "z": round(norm_lat(PDN_LAT), 3),
         "y": round(norm_z(3070), 3) + 1.5, "label": "⛰ Piton des Neiges  3070 m",       "c": "#4fc3f7"},
    ]
    markers_json = json.dumps(markers)

    synth_warning = "⚠️ Terrain synthétique — lance tools/build_dem.py" if synthetic else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0d0d18; overflow:hidden; font-family:'Segoe UI',sans-serif; }}
#c {{ display:block; width:100%; height:100%; }}
#legend {{
    position:absolute; bottom:12px; left:12px;
    background:rgba(255,255,255,0.92); border-radius:8px;
    padding:8px 12px; font-size:11px; color:#333;
    border:1px solid #ccc; box-shadow:0 2px 8px rgba(0,0,0,0.15);
    pointer-events:none;
}}
#legend b {{ color:#1a1a2e; display:block; margin-bottom:4px; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:middle; }}
#tooltip {{
    position:absolute; display:none; pointer-events:none;
    background:rgba(255,255,255,0.95); border-radius:6px;
    padding:6px 10px; font-size:11px; color:#333;
    border:1px solid #ccc; box-shadow:0 2px 8px rgba(0,0,0,0.2);
    max-width:220px;
}}
#toolbar {{
    position:absolute; top:10px; right:10px;
    display:flex; flex-direction:column; gap:6px;
}}
.tbtn {{
    background:rgba(255,255,255,0.92); border:1px solid #ccc;
    border-radius:7px; padding:7px 13px; font-size:12px; font-weight:600;
    color:#555; cursor:pointer; text-align:center;
    box-shadow:0 2px 6px rgba(0,0,0,0.12); user-select:none;
    transition: background 0.15s, color 0.15s;
}}
.tbtn.active {{ background:#1a1a2e; color:#fff; border-color:#1a1a2e; }}
.tbtn:hover:not(.active) {{ background:#f0f0f0; }}
#hint {{
    background:rgba(255,255,255,0.88); border-radius:7px;
    padding:6px 12px; font-size:10px; color:#666;
    border:1px solid #ddd; text-align:center; line-height:1.7;
}}
#synth {{ position:absolute; top:10px; left:10px; background:rgba(255,200,0,0.9);
    padding:4px 10px; border-radius:6px; font-size:11px; font-weight:600;
    display:{'block' if synth_warning else 'none'}; }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="legend">
  <b>Magnitude</b>
  <span class="dot" style="background:#a0a0a0"></span>&lt;1 &nbsp;
  <span class="dot" style="background:#4fc3f7"></span>1–2 &nbsp;
  <span class="dot" style="background:#81c784"></span>2–3<br>
  <span class="dot" style="background:#fb8c00"></span>3–4 &nbsp;
  <span class="dot" style="background:#ef5350"></span>4–5 &nbsp;
  <span class="dot" style="background:#b71c1c"></span>≥5
</div>
<div id="tooltip"></div>
<div id="toolbar">
  <button class="tbtn active" id="btn-rot"  onclick="setMode('rotate')">🔄 Rotation</button>
  <button class="tbtn"        id="btn-pan"  onclick="setMode('pan')">✋ Translation</button>
  <button class="tbtn"        id="btn-rst"  onclick="resetView()">⌂ Réinitialiser</button>
  <div id="hint">Molette : zoom</div>
</div>
<div id="synth">{synth_warning}</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const ROWS={nrows}, COLS={ncols};
const HEIGHTS={heights_json};
const QUAKES={quakes_json};
const MARKERS={markers_json};

let scene, camera, renderer, pivotGroup;
let isDragging=false;
let mode='rotate';   // 'rotate' | 'pan'
let lastX=0, lastY=0;
const tooltip = document.getElementById('tooltip');

function setMode(m) {{
    mode = m;
    document.getElementById('btn-rot').classList.toggle('active', m==='rotate');
    document.getElementById('btn-pan').classList.toggle('active', m==='pan');
}}
function resetView() {{
    pivotGroup.rotation.set(0,0,0);
    pivotGroup.position.set(0,0,0);
    camera.position.set(0,55,95);
    camera.lookAt(0,0,0);
}}

// ── Scène ──────────────────────────────────────────────────────────────────
scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d18);
scene.fog = new THREE.Fog(0x0d0d18, 150, 350);

camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
camera.position.set(0, 55, 95);
camera.lookAt(0, 0, 0);

renderer = new THREE.WebGLRenderer({{antialias:true, canvas:document.getElementById('c')}});
renderer.setPixelRatio(window.devicePixelRatio);

// Lumières
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xfffbe0, 0.9);
sun.position.set(40, 80, 40);
scene.add(sun);

pivotGroup = new THREE.Group();
scene.add(pivotGroup);

// ── Terrain ────────────────────────────────────────────────────────────────
(function buildTerrain() {{
    const geo = new THREE.BufferGeometry();
    const verts=[], colors=[], indices=[];

    for(let r=0;r<ROWS;r++) for(let c=0;c<COLS;c++) {{
        const x = (c/(COLS-1)-0.5)*60;
        const z = (r/(ROWS-1)-0.5)*60;
        const h = HEIGHTS[r][c];
        verts.push(x,h,z);
        // Couleur hypsométrique
        const t = Math.max(0,Math.min(1, h / 8));
        let rr,gg,bb;
        if(t<0.05)      {{ rr=0.1; gg=0.25; bb=0.5; }}
        else if(t<0.25) {{ rr=0.18; gg=0.55; bb=0.3; }}
        else if(t<0.50) {{ rr=0.6; gg=0.72; bb=0.25; }}
        else if(t<0.70) {{ rr=0.8; gg=0.5; bb=0.17; }}
        else if(t<0.88) {{ rr=0.7; gg=0.22; bb=0.13; }}
        else            {{ rr=0.94; gg=0.93; bb=0.92; }}
        colors.push(rr,gg,bb);
    }}
    for(let r=0;r<ROWS-1;r++) for(let c=0;c<COLS-1;c++) {{
        const a=r*COLS+c, b=a+1, d=a+COLS, e=d+1;
        indices.push(a,d,b, b,d,e);
    }}
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts,3));
    geo.setAttribute('color',    new THREE.Float32BufferAttribute(colors,3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({{vertexColors:true, side:THREE.FrontSide, transparent:true, opacity:0.88}});
    pivotGroup.add(new THREE.Mesh(geo,mat));
}})();

// ── Séismes ────────────────────────────────────────────────────────────────
const quakeMeshes = [];
QUAKES.forEach(q => {{
    const geo = new THREE.SphereGeometry(q.r, 8, 8);
    const mat = new THREE.MeshLambertMaterial({{color: q.c}});
    const m = new THREE.Mesh(geo, mat);
    m.position.set(q.x, q.y, q.z);
    m.userData.label = q.label;
    pivotGroup.add(m);
    quakeMeshes.push(m);
}});

// ── Marqueurs sommets ──────────────────────────────────────────────────────
MARKERS.forEach(mk => {{
    const geo = new THREE.OctahedronGeometry(0.8);
    const mat = new THREE.MeshLambertMaterial({{color: mk.c}});
    const m = new THREE.Mesh(geo,mat);
    m.position.set(mk.x, mk.y, mk.z);
    m.userData.label = mk.label;
    pivotGroup.add(m);
    quakeMeshes.push(m);
}});

// ── Contrôles souris ───────────────────────────────────────────────────────
const canvas = renderer.domElement;
const raycaster = new THREE.Raycaster();
raycaster.params.Mesh = {{threshold:0.5}};
const mouse2d = new THREE.Vector2();

canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('mousedown', e => {{
    isDragging=true; lastX=e.clientX; lastY=e.clientY;
}});
canvas.addEventListener('mousemove', e => {{
    // Tooltip au survol
    const rect = canvas.getBoundingClientRect();
    mouse2d.x = ((e.clientX-rect.left)/rect.width)*2-1;
    mouse2d.y = -((e.clientY-rect.top)/rect.height)*2+1;
    raycaster.setFromCamera(mouse2d, camera);
    const hits = raycaster.intersectObjects(quakeMeshes);
    if(hits.length>0 && hits[0].object.userData.label) {{
        tooltip.style.display='block';
        tooltip.style.left=(e.clientX+12)+'px';
        tooltip.style.top=(e.clientY-10)+'px';
        tooltip.textContent=hits[0].object.userData.label;
    }} else {{
        tooltip.style.display='none';
    }}
    if(!isDragging) return;
    const dx=e.clientX-lastX, dy=e.clientY-lastY;
    lastX=e.clientX; lastY=e.clientY;
    if(mode==='pan') {{
        pivotGroup.position.x += dx*0.08;
        pivotGroup.position.y -= dy*0.08;
    }} else {{
        pivotGroup.rotation.y += dx*0.008;
        pivotGroup.rotation.x += dy*0.008;
    }}
}});
canvas.addEventListener('mouseup',    ()=>{{ isDragging=false; }});
canvas.addEventListener('mouseleave', ()=>{{ isDragging=false; tooltip.style.display='none'; }});
canvas.addEventListener('wheel', e=>{{
    e.preventDefault();
    camera.position.z = Math.max(10, Math.min(250, camera.position.z + e.deltaY*0.08));
}}, {{passive:false}});
canvas.addEventListener('dblclick', resetView);

// Touch
let lastTouchDist=0;
canvas.addEventListener('touchstart', e=>{{
    if(e.touches.length===1){{ isDragging=true; lastX=e.touches[0].clientX; lastY=e.touches[0].clientY; }}
    if(e.touches.length===2){{ lastTouchDist=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY); }}
}});
canvas.addEventListener('touchmove', e=>{{
    e.preventDefault();
    if(e.touches.length===1 && isDragging){{
        const dx=e.touches[0].clientX-lastX, dy=e.touches[0].clientY-lastY;
        lastX=e.touches[0].clientX; lastY=e.touches[0].clientY;
        pivotGroup.rotation.y+=dx*0.008; pivotGroup.rotation.x+=dy*0.008;
    }}
    if(e.touches.length===2){{
        const d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);
        camera.position.z=Math.max(10,Math.min(250,camera.position.z-(d-lastTouchDist)*0.2));
        lastTouchDist=d;
    }}
}},{{passive:false}});
canvas.addEventListener('touchend',()=>isDragging=false);

// ── Resize & boucle ────────────────────────────────────────────────────────
function resize() {{
    const w=canvas.clientWidth, h=canvas.clientHeight||600;
    renderer.setSize(w,h,false);
    camera.aspect=w/h;
    camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resize);
resize();

(function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}})();
</script>
</body></html>"""
    return html
