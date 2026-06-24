import streamlit as st
import numpy as np
import trimesh
import json
import streamlit.components.v1 as components
from core.tpms_lattice import generate_multi_field_lattice, LATTICE_LIBRARY
from core.physics import calculate_multi_pressure_field
from core.geometry import load_and_voxelize_mesh

if "clicks" not in st.session_state:
    st.session_state.clicks = []

st.set_page_config(
    page_title="Generative Stress-Lattice App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dashboard UI injection
st.markdown("""
    <style>
    .main-title { font-size: 42px !important; font-weight: 800 !important; color: #00ffcc; margin-bottom: 5px; }
    .subtitle { font-size: 16px !important; color: #a0aec0; margin-bottom: 30px; }
    .card { background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 5px solid #00ffcc; margin-bottom: 20px; }
    .coordinate-badge { background-color: #2d3748; color: #ff3333; padding: 4px 8px; border-radius: 5px; font-family: monospace; margin: 4px; display: inline-block; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Stress-Driven Generative Lattice Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Synthesize adaptive, topology-optimized internal microstructures on-the-fly.</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### 🎛️ 1. Studio Operation Mode")
view_mode = st.sidebar.radio(
    "Select Interface Focus Mode:",
    ["Mode A: View Original STL (Paint Stress Nodes)", "Mode B: View Generated Microstructure Matrix"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 2. Global Lattice Parameters")
resolution = st.sidebar.slider("Voxel Grid Resolution", 32, 96, 64, step=16)
periods = st.sidebar.slider("Lattice Cell Frequency (Periods)", 1.0, 10.0, 4.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧮 3. Mathematical Topologies")
options = list(LATTICE_LIBRARY.keys())
base_style = st.sidebar.selectbox("Baseline Structure (Low Pressure)", options, index=0)
press_style = st.sidebar.selectbox("Reinforcement Structure (High Pressure)", options, index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 4. Density Configurations")
base_dense = st.sidebar.slider("Baseline Wall Thickness", -0.4, 0.4, 0.0, step=0.05)
max_dense = st.sidebar.slider("Max Pressure Thickness", -0.4, 0.6, 0.3, step=0.05)
inf_radius = st.sidebar.slider("Pressure Blend Radius (Bubble Size)", 0.1, 1.5, 0.4, step=0.05)

# --- WORKSPACE ---
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="card"><h3>📦 Interactive Web Viewport Canvas</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Target Part File (STL/OBJ format)", type=["stl", "obj"], label_visibility="collapsed")
    
    if uploaded_file:
        input_path = f"data/input/{uploaded_file.name}"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Processing live graphics display context..."):
            cad_mesh, _ = load_and_voxelize_mesh(input_path, resolution=resolution)
            
            # Center and scale to normalized bounds (-1 to 1) 
            centroid = cad_mesh.vertices.mean(axis=0)
            min_bounds = cad_mesh.vertices.min(axis=0)
            max_bounds = cad_mesh.vertices.max(axis=0)
            extents = max_bounds - min_bounds
            
            scale_factor = 2.0 / np.max(extents) if np.max(extents) > 0 else 1.0
            norm_vertices = (cad_mesh.vertices - centroid) * scale_factor
            geom_data = {"vertices": norm_vertices.tolist(), "faces": cad_mesh.faces.tolist()}
            
            lattice_data = None
            if "Mode B" in view_mode:
                master_field = calculate_multi_pressure_field(
                    resolution=resolution, click_list=st.session_state.clicks, influence_radius=inf_radius
                )
                lattice_mesh = generate_multi_field_lattice(
                    resolution=resolution, periods=periods, base_style=base_style, custom_base_eq="",
                    pressure_style=press_style, custom_press_eq="", base_thickness=base_dense,
                    max_pressure_thickness=max_dense, combined_pressure_field=master_field
                )
                
                if lattice_mesh is not None:
                    comps = lattice_mesh.split(only_watertight=False)
                    if len(comps) > 1:
                        lattice_mesh = max(comps, key=lambda m: m.area)
                    
                    # --- UNIFORM PROPORTIONAL FITTING ENGINE ---
                    # 1. Zero-out the lattice to its own true local center
                    lat_centroid = lattice_mesh.vertices.mean(axis=0)
                    lat_centered = lattice_mesh.vertices - lat_centroid
                    
                    # 2. Get bounding extents for both systems
                    lat_extents = lat_centered.max(axis=0) - lat_centered.min(axis=0)
                    stl_norm_extents = norm_vertices.max(axis=0) - norm_vertices.min(axis=0)
                    
                    # 3. Compute proportional aspect ratios on each axis
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratios = stl_norm_extents / lat_extents
                        # Safe fallback filtering out invalid entries
                        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
                    
                    # 4. Use the MINIMUM ratio to fit perfectly inside the smallest dimension of the shell
                    uniform_ratio = np.min(ratios) if len(ratios) > 0 else 1.0
                    
                    # 5. Proportionally scale and move it directly to the normalized STL center point
                    lat_vertices = lat_centered * uniform_ratio
                    
                    lattice_data = {"vertices": lat_vertices.tolist(), "faces": lattice_mesh.faces.tolist()}

            # Map coordinates safely into the normalized viewport scale domain
            normalized_clicks = []
            for pt in st.session_state.clicks:
                norm_pt = (np.array(pt) - centroid) * scale_factor
                normalized_clicks.append(norm_pt.tolist())

            # --- RECEIVE CLICK EVENTS FROM JAVASCRIPT ---
            query_params = st.query_params
            if "incoming_x" in query_params:
                try:
                    click_pt = [
                        float(query_params["incoming_x"]),
                        float(query_params["incoming_y"]),
                        float(query_params["incoming_z"])
                    ]
                    world_pt = (np.array(click_pt) / scale_factor) + centroid
                    world_pt_list = world_pt.tolist()
                    if world_pt_list not in st.session_state.clicks:
                        st.session_state.clicks.append(world_pt_list)
                    st.query_params.clear()
                    st.rerun()
                except Exception:
                    pass

            # --- THREE.JS GRAPHICS COMPONENT ENGINE ---
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
                <style>
                    body {{ margin: 0; background-color: #0f1115; overflow: hidden; }}
                    #canvas-container {{ width: 100vw; height: 100vh; }}
                    #overlay-ui {{ position: absolute; top: 10px; left: 10px; color: #00ffcc; background: rgba(30,30,30,0.85); padding: 8px 12px; border-radius: 5px; pointer-events: none; font-size: 13px; font-weight: bold; border-left: 3px solid #00ffcc; }}
                </style>
            </head>
            <body>
                <div id="overlay-ui">Active Canvas Mode: {"Mode A (Click Shell to Add Nodes)" if "Mode A" in view_mode else "Mode B (Lattice View)"}</div>
                <div id="canvas-container"></div>
                <script>
                    const container = document.getElementById('canvas-container');
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color('#0f1115');

                    const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 100);
                    camera.position.set(3, 2, 4);

                    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    renderer.setPixelRatio(window.devicePixelRatio);
                    container.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;

                    const grid = new THREE.GridHelper(6, 30, '#00ffcc', '#222630');
                    grid.position.y = -1.2;
                    scene.add(grid);

                    scene.add(new THREE.AmbientLight('#ffffff', 0.5));
                    const topLight = new THREE.DirectionalLight('#ffffff', 0.8);
                    topLight.position.set(5, 8, 5);
                    scene.add(topLight);

                    const sideLight = new THREE.DirectionalLight('#ffffff', 0.5);
                    sideLight.position.set(-5, 3, -5);
                    scene.add(sideLight);

                    const clickList = {json.dumps(normalized_clicks)};
                    const infRadius = {inf_radius * scale_factor};

                    clickList.forEach(pt => {{
                        const coreMesh = new THREE.Mesh(
                            new THREE.SphereGeometry(0.04, 16, 16),
                            new THREE.MeshBasicMaterial({{ color: 0xff3344 }})
                        );
                        coreMesh.position.set(pt[0], pt[1], pt[2]);
                        scene.add(coreMesh);

                        const bubMesh = new THREE.Mesh(
                            new THREE.SphereGeometry(infRadius, 32, 32),
                            new THREE.MeshBasicMaterial({{ color: 0xff3344, transparent: true, opacity: 0.15 }})
                        );
                        bubMesh.position.set(pt[0], pt[1], pt[2]);
                        scene.add(bubMesh);
                    }});

                    function buildMeshGeometry(data) {{
                        const geometry = new THREE.BufferGeometry();
                        const vertices = [];
                        data.faces.forEach(face => {{
                            face.forEach(vIdx => {{
                                vertices.push(data.vertices[vIdx][0], data.vertices[vIdx][1], data.vertices[vIdx][2]);
                            }});
                        }});
                        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                        geometry.computeVertexNormals();
                        return geometry;
                    }}

                    const baseGeomData = {json.dumps(geom_data)};
                    const baseGeometry = buildMeshGeometry(baseGeomData);

                    if ("{"Mode A" in view_mode}" === "True") {{
                        const mat = new THREE.MeshStandardMaterial({{ color: 0x546e7a, roughness: 0.4, metalness: 0.2, side: THREE.DoubleSide }});
                        const mesh = new THREE.Mesh(baseGeometry, mat);
                        scene.add(mesh);

                        const raycaster = new THREE.Raycaster();
                        const mouse = new THREE.Vector2();
                        
                        let mDownTime = 0;
                        window.addEventListener('mousedown', () => {{ mDownTime = Date.now(); }});

                        window.addEventListener('mouseup', (e) => {{
                            if (Date.now() - mDownTime > 200) return; 

                            mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
                            mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
                            raycaster.setFromCamera(mouse, camera);
                            const intersects = raycaster.intersectObject(mesh);
                            
                            if(intersects.length > 0) {{
                                const p = intersects[0].point;
                                const parentUrl = new URL(window.parent.location.href);
                                parentUrl.searchParams.set('incoming_x', p.x.toFixed(4));
                                parentUrl.searchParams.set('incoming_y', p.y.toFixed(4));
                                parentUrl.searchParams.set('incoming_z', p.z.toFixed(4));
                                window.parent.location.href = parentUrl.toString();
                            }}
                        }});
                    }} else {{
                        const latGeomData = {json.dumps(lattice_data) if lattice_data else "null"};
                        if(latGeomData) {{
                            const latGeom = buildMeshGeometry(latGeomData);
                            const latMat = new THREE.MeshStandardMaterial({{ color: 0xb0bec5, roughness: 0.3, metalness: 0.7, side: THREE.DoubleSide }});
                            const latMesh = new THREE.Mesh(latGeom, latMat);
                            scene.add(latMesh);
                        }}
                        
                        const wireframe = new THREE.WireframeGeometry(baseGeometry);
                        const line = new THREE.LineSegments(wireframe);
                        line.material.color.setHex(0xff3344);
                        line.material.opacity = 0.35;
                        line.material.transparent = true;
                        scene.add(line);
                    }}

                    function animate() {{
                        requestAnimationFrame(animate);
                        controls.update();
                        renderer.render(scene, camera);
                    }}
                    animate();

                    window.addEventListener('resize', () => {{
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, window.innerHeight);
                    }});
                </script>
            </body>
            </html>
            """
            
            components.html(html_template, height=600, scrolling=False)

with col_right:
    st.markdown('<div class="card"><h3>🎯 Applied Stress Fields</h3></div>', unsafe_allow_html=True)
    
    if not st.session_state.clicks:
        st.info("No stress vectors applied yet. Select Mode A and click directly on the component structure surface layout.")
    else:
        st.write(f"**Total Registered Nodes:** {len(st.session_state.clicks)}")
        for idx, click in enumerate(st.session_state.clicks):
            st.markdown(f"**Node {idx+1}:** <span class='coordinate-badge'>X: {click[0]:.2f}, Y: {click[1]:.2f}, Z: {click[2]:.2f}</span>", unsafe_allow_html=True)
            
        st.write("")
        if st.button("Reset All Coordinates", type="secondary", use_container_width=True):
            st.session_state.clicks = []
            st.query_params.clear()
            st.rerun()