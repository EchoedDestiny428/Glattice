import streamlit as st
import numpy as np
import json
import streamlit.components.v1 as components

from core.geometry import load_and_voxelize_mesh
from core.physics import calculate_multi_pressure_field
from core.tpms_lattice import generate_multi_field_lattice, LATTICE_LIBRARY

import tempfile, os

@st.cache_resource
def _make_listener():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, 'index.html'), 'w') as f:
        f.write("""<!DOCTYPE html><html><body><script>
window.parent.postMessage({type:"streamlit:componentReady",apiVersion:1},"*");
window.parent.postMessage({type:"streamlit:setFrameHeight",height:0},"*");
new BroadcastChannel("lattice-clicks").onmessage = e => {
    window.parent.postMessage({
        type:"streamlit:setComponentValue",
        value:e.data,
        dataType:"json"
    },"*");
};
</script></body></html>""")
    return components.declare_component("lattice_click_listener", path=d)

_click_listener = _make_listener()

st.set_page_config(
    page_title="Generative Stress-Lattice App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# STATE
# -----------------------------
if "clicks" not in st.session_state:
    st.session_state.clicks = []

@st.cache_resource(show_spinner=False)
def cached_mesh(path, resolution):
    return load_and_voxelize_mesh(path, resolution)


@st.cache_data(show_spinner=False)
def cached_pressure(resolution, clicks_tuple, radius):
    clicks = [list(c) for c in clicks_tuple]
    return calculate_multi_pressure_field(
        resolution=resolution,
        click_list=clicks,
        influence_radius=radius
    )

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #00ffcc;
}
.subtitle {
    font-size: 14px;
    color: #a0aec0;
}
.card {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.coordinate-badge {
    background-color: #2d3748;
    color: #ff4444;
    padding: 3px 6px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Stress-Lattice Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Topology-driven adaptive lattice generation</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

view_mode = st.sidebar.radio(
    "Mode",
    [
        "Mode A: View CAD (Click Nodes)",
        "Mode B: View Lattice"
    ]
)

resolution = st.sidebar.slider("Resolution", 32, 96, 64, step=16)
periods = st.sidebar.slider("Periods", 1.0, 10.0, 4.0, step=0.5)

base_style = st.sidebar.selectbox(
    "Base TPMS",
    list(LATTICE_LIBRARY.keys())
)

press_style = st.sidebar.selectbox(
    "Pressure TPMS",
    list(LATTICE_LIBRARY.keys())
)

density_base = st.sidebar.slider(
    "Base Density",
    0.0, 1.0, 0.5, step=0.05
)

density_max = st.sidebar.slider(
    "Max Pressure Density",
    0.0, 1.0, 0.8, step=0.05
)
stress_radius = st.sidebar.slider("Influence Radius", 0.1, 1.5, 0.4)

col1, col2 = st.columns([3, 2])

with col1:

    uploaded = st.file_uploader(
        "Upload STL/OBJ",
        type=["stl", "obj"]
    )

    if uploaded:

        path = f"data/{uploaded.name}"

        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        cad_mesh, cad_mask, meta = cached_mesh(
            path,
            resolution
        )

        # -----------------------------
        # NORMALIZED MESH (NO MORE SCALE)
        # -----------------------------
        geom_data = {
            "vertices": cad_mesh.vertices.tolist(),
            "faces": cad_mesh.faces.tolist()
        }

                # -----------------------------
        # CLICK LIST (ALREADY NORMALIZED)
        # -----------------------------
        clicks = st.session_state.clicks

        clicks_tuple = tuple(
            tuple(c) for c in clicks
        )

        pressure_field = cached_pressure(
            resolution,
            clicks_tuple,
            stress_radius
        )

        lattice_mesh = None
        lattice_data = None

        if view_mode == "Mode B: View Lattice":

            lattice_mesh = generate_multi_field_lattice(
                resolution=resolution,
                periods=periods,
                base_style=base_style,
                pressure_style=press_style,
                density_base=density_base,
                density_max=density_max,
                combined_pressure_field=pressure_field
            )

            if lattice_mesh is not None:

                try:
                    lattice_mesh = lattice_mesh.intersection(
                        cad_mesh,
                        engine="manifold"
                    )
                except Exception as e:
                    st.warning(f"Boolean skipped: {e}")

                if lattice_mesh and len(lattice_mesh.vertices) > 0:

                    comps = lattice_mesh.split(
                        only_watertight=False
                    )

                    if comps:
                        lattice_mesh = max(
                            comps,
                            key=lambda m: m.area
                        )

                    lattice_data = {
                        "vertices": lattice_mesh.vertices.tolist(),
                        "faces": lattice_mesh.faces.tolist()
                    }
                else:
                    lattice_data = None

        # -----------------------------
        # CLICK HANDLING
        # -----------------------------
        query_params = st.query_params

        click_data = _click_listener(key="lc", default=None)
        if click_data and click_data.get('t') != st.session_state.get('_lt'):
            st.session_state['_lt'] = click_data['t']
            pt = [click_data['x'], click_data['y'], click_data['z']]
            if not any(np.allclose(pt, c, atol=1e-3) for c in st.session_state.clicks):
                st.session_state.clicks.append(pt)
            st.rerun()
                
        normalized_clicks = st.session_state.clicks

        html = f"""
        <html>
        <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

        <style>
            body {{ margin:0; overflow:hidden; background:#0f1115; }}
        </style>
        </head>

        <body>
        <div id="c"></div>

        <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#111318');

        scene.add(new THREE.AmbientLight(0xffffff, 0.75));

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
        keyLight.position.set(5, 10, 7);
        scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
        fillLight.position.set(-5, 3, -2);
        scene.add(fillLight);

        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
        camera.position.set(3,2,4);

        const renderer = new THREE.WebGLRenderer({{antialias:true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);

        const clicks = {json.dumps(normalized_clicks)};

        clicks.forEach(p => {{
            const core = new THREE.Mesh(
                new THREE.SphereGeometry(0.03, 16, 16),
                new THREE.MeshStandardMaterial({{
                    color: 0xff3344,
                    roughness: 0.4,
                    metalness: 0.2
                }})
            );

            core.position.set(p[0], p[1], p[2]);
            scene.add(core);

            const glow = new THREE.Mesh(
                new THREE.SphereGeometry(0.06, 24, 24),
                new THREE.MeshBasicMaterial({{
                    color: 0xff3344,
                    transparent: true,
                    opacity: 0.15
                }})
            );

            glow.position.set(p[0], p[1], p[2]);
            scene.add(glow);

        }});

        function build(data) {{
            const g = new THREE.BufferGeometry();
            const v = [];

            data.faces.forEach(f => {{
                f.forEach(i => {{
                    v.push(
                        data.vertices[i][0],
                        data.vertices[i][1],
                        data.vertices[i][2]
                    );
                }});
            }});

            g.setAttribute(
                'position',
                new THREE.Float32BufferAttribute(v,3)
            );

            g.computeVertexNormals();
            return g;
        }}

        const base = build({json.dumps(geom_data)});

        if ("{view_mode}" === "Mode A: View CAD (Click Nodes)") {{
            const cadMesh = new THREE.Mesh(
                base,
                new THREE.MeshStandardMaterial({{
                    color:0x546e7a,
                    side:THREE.DoubleSide
                }})
            );
            scene.add(cadMesh);

            // ── Visible debug overlay (no devtools needed) ─────────────
            const dbg = document.createElement('div');
            dbg.style.cssText = `
                position:fixed; top:10px; left:10px; z-index:999;
                background:rgba(0,0,0,0.75); color:#facc15;
                font:12px monospace; padding:8px 12px; border-radius:6px;
                pointer-events:none; white-space:pre;
            `;
            dbg.innerText = 'Move mouse over mesh, then press P';
            document.body.appendChild(dbg);

            // ── Make canvas capture keyboard ───────────────────────────
            renderer.domElement.setAttribute('tabindex', '0');
            renderer.domElement.style.outline = 'none';

            let lastHit = null;

            renderer.domElement.addEventListener('mousemove', e => {{
                renderer.domElement.focus();   // <-- critical: grabs keyboard focus

                const rect = renderer.domElement.getBoundingClientRect();
                const ndc = new THREE.Vector2(
                    ((e.clientX - rect.left) / rect.width)  * 2 - 1,
                    -((e.clientY - rect.top)  / rect.height) * 2 + 1
                );

                const ray = new THREE.Raycaster();
                ray.setFromCamera(ndc, camera);
                const hits = ray.intersectObject(cadMesh);

                if (hits.length > 0) {{
                    lastHit = hits[0].point;
                    renderer.domElement.style.cursor = 'crosshair';
                    dbg.style.color = '#4ade80';
                    dbg.innerText =
                        '✓ Hovering — press P to place node\\n' +
                        'x: ' + lastHit.x.toFixed(4) + '\\n' +
                        'y: ' + lastHit.y.toFixed(4) + '\\n' +
                        'z: ' + lastHit.z.toFixed(4);
                }} else {{
                    lastHit = null;
                    renderer.domElement.style.cursor = 'default';
                    dbg.style.color = '#facc15';
                    dbg.innerText = 'Move mouse over mesh, then press P';
                }}
            }});

            // ── keydown on canvas (not window) ─────────────────────────
            renderer.domElement.addEventListener('keydown', e => {{
                dbg.style.color = '#38bdf8';
                dbg.innerText = 'Key detected: ' + e.key;   // shows ANY key press

                if (e.key !== 'p' && e.key !== 'P') return;

                if (!lastHit) {{
                    dbg.style.color = '#f87171';
                    dbg.innerText = '✗ P pressed but cursor is off the mesh';
                    return;
                }}

                dbg.style.color = '#a78bfa';
                dbg.innerText = 'Sending node to Streamlit...\\n' +
                    lastHit.x.toFixed(4) + ', ' +
                    lastHit.y.toFixed(4) + ', ' +
                    lastHit.z.toFixed(4);

                new BroadcastChannel('lattice-clicks').postMessage({{
                    x: lastHit.x, y: lastHit.y, z: lastHit.z, t: Date.now()
                }});
                dbg.style.color = '#a78bfa';
                dbg.innerText = 'Node sent!\\n' +
                    lastHit.x.toFixed(4) + ', ' +
                    lastHit.y.toFixed(4) + ', ' +
                    lastHit.z.toFixed(4);
            }});

        }} else {{

            const lat = {json.dumps(lattice_data) if 'lattice_data' in locals() and lattice_data else "null"};

            if (lat) {{
                const g = build(lat);
                const m = new THREE.Mesh(
                    g,
                    new THREE.MeshStandardMaterial({{
                        color:0xb0bec5,
                        side:THREE.DoubleSide
                    }})
                );
                scene.add(m);
            }}

            const wire = new THREE.LineSegments(
                new THREE.WireframeGeometry(base)
            );

            wire.material.color.setHex(0xff3344);
            scene.add(wire);
        }}

        function animate(){{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene,camera);
        }}

        animate();
        </script>
        </body>
        </html>
        """

        components.html(html, height=650)

with col2:

    st.markdown("### Stress Nodes")

    if not st.session_state.clicks:
        st.info("Click on CAD in Mode A")

    else:
        for i, c in enumerate(st.session_state.clicks):
            st.markdown(
                f"Node {i+1}: "
                f"<span class='coordinate-badge'>"
                f"{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}"
                f"</span>",
                unsafe_allow_html=True
            )

        if st.button("Reset"):
            st.session_state.clicks = []
            st.rerun()