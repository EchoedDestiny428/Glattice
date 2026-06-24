import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import trimesh
import pyvista as pv
from core.tpms_lattice import generate_multi_field_lattice, LATTICE_LIBRARY
from core.physics import calculate_multi_pressure_field
from core.geometry import load_and_voxelize_mesh

if "clicks" not in st.session_state:
    st.session_state.clicks = []

st.set_page_config(page_title="Generative Stress-Lattice App", layout="wide")
st.title("Stress-Driven Generative Lattice Studio")
st.write("Upload a CAD body, define high-pressure locations, and generate seamless morphing structures.")

st.sidebar.header("1. Global Lattice Parameters")
resolution = st.sidebar.slider("Voxel Grid Resolution", 32, 96, 64, step=16)
periods = st.sidebar.slider("Lattice Cell Frequency (Periods)", 1.0, 10.0, 4.0, step=0.5)

options = list(LATTICE_LIBRARY.keys()) + ["Custom Equation"]
base_style = st.sidebar.selectbox("Baseline Structure (Low Pressure Area)", options, index=0)
custom_base = ""
if base_style == "Custom Equation":
    custom_base = st.sidebar.text_input("Enter Custom Equation (Base)", "np.sin(X)*np.cos(Y)")

press_style = st.sidebar.selectbox("Reinforcement Structure (High Pressure Area)", options, index=1)
custom_press = ""
if press_style == "Custom Equation":
    custom_press = st.sidebar.text_input("Enter Custom Equation (Pressure)", "np.sin(X)*np.sin(Y)*np.sin(Z)")

st.sidebar.header("2. Density & Profile Configurations")
base_dense = st.sidebar.slider("Baseline Wall Thickness", -0.4, 0.4, 0.0, step=0.05)
max_dense = st.sidebar.slider("Max Pressure Wall Thickness", -0.4, 0.6, 0.3, step=0.05)
inf_radius = st.sidebar.slider("Pressure Blend Radius (Influence Field)", 0.1, 1.5, 0.4, step=0.05)

tab1, tab2 = st.tabs(["Geometry Input & Stress Painting", "Lattice Synthesis & 3D Preview"])

with tab1:
    st.subheader("CAD Boundary Selection")
    uploaded_file = st.file_uploader("Upload Target Part File (STL/OBJ format)", type=["stl", "obj"])
    
    if uploaded_file:
        input_path = f"data/input/{uploaded_file.name}"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Loaded: {uploaded_file.name}")
        
        st.subheader("Define High-Pressure Nodes")
        col1, col2, col3 = st.columns(3)
        with col1: px = st.number_input("Pressure Coordinate X (-1.0 to 1.0)", -1.0, 1.0, 0.0, step=0.1)
        with col2: py = st.number_input("Pressure Coordinate Y (-1.0 to 1.0)", -1.0, 1.0, 0.0, step=0.1)
        with col3: pz = st.number_input("Pressure Coordinate Z (-1.0 to 1.0)", -1.0, 1.0, 0.0, step=0.1)
        
        if st.button("Add Selected Coordinate to Pressure List"):
            st.session_state.clicks.append((px, py, pz))
            
        if st.session_state.clicks:
            st.write("**Active High-Pressure Points Applied:**")
            st.write(st.session_state.clicks)
            if st.button("Reset All Coordinates"):
                st.session_state.clicks = []
                st.rerun()

with tab2:
    st.subheader("Synthesis & Visualization")
    if not uploaded_file:
        st.info("Please complete Step 1 by uploading a baseline CAD shell profile geometry.")
    else:
        if st.button("Execute Generative Morphing Engine", type="primary"):
            with st.spinner("Synthesizing multi-field mathematical boundaries..."):
                
                input_path = f"data/input/{uploaded_file.name}"
                
                master_field = calculate_multi_pressure_field(
                    resolution=resolution, 
                    click_list=st.session_state.clicks, 
                    influence_radius=inf_radius
                )
                
                cad_mesh, cad_mask = load_and_voxelize_mesh(input_path, resolution=resolution)
                
                if not cad_mesh.is_watertight:
                    st.warning("⚠️ Warning: Your uploaded CAD mesh has open holes or non-manifold edges. The engine is attempting an automated shrink-wrap repair.")
                
                lattice_mesh = generate_multi_field_lattice(
                    resolution=resolution,
                    periods=periods,
                    base_style=base_style,
                    custom_base_eq=custom_base,
                    pressure_style=press_style,
                    custom_press_eq=custom_press,
                    base_thickness=base_dense,
                    max_pressure_thickness=max_dense,
                    combined_pressure_field=master_field
                )
                
                if lattice_mesh is not None:
                    out_path = f"data/output/reinforced_{uploaded_file.name}"
                    lattice_mesh.export(out_path)
                    
                    st.balloons()
                    st.success("Lattice matrix compiled cleanly into a single unified topology structure!")
                    
                    st.subheader("Interactive 3D Preview")
                    
                    # Convert our generated trimesh to PyVista PolyData
                    pv_mesh = pv.PolyData(lattice_mesh.vertices, np.c_[np.full(len(lattice_mesh.faces), 3), lattice_mesh.faces])  # type: ignore
                    
                    # Initialize the PyVista offscreen renderer
                    plotter = pv.Plotter(window_size=[800, 500], off_screen=True)  # type: ignore
                    plotter.set_background("#1e1e1e")  # type: ignore
                    plotter.add_mesh(pv_mesh, color="#00ffcc", show_edges=True, edge_color="#003322", smooth_shading=True)  # type: ignore
                    plotter.add_axes()  # type: ignore
                    plotter.view_isometric()  # type: ignore
                    
                    # Export the renderer's scene to a standalone inline HTML block
                    html_path = "data/output/preview.html"
                    plotter.export_html(html_path, backend="trame")  # type: ignore
                    
                    # Load and inject the HTML directly into the web canvas safely
                    with open(html_path, "r", encoding="utf-8") as html_file:
                        render_html = html_file.read()
                    
                    components.html(render_html, height=500, scrolling=False)
                    
                    with open(out_path, "rb") as file:
                        st.download_button(
                            label="Download Reinforced Part STL",
                            data=file,
                            file_name=f"reinforced_{uploaded_file.name}",
                            mime="application/sla"
                        )
                else:
                    st.error("The parameters selected resolved to an empty mathematical intersection set.")