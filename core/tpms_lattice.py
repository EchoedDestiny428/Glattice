import numpy as np
import trimesh
from skimage import measure


LATTICE_LIBRARY = {
    "Gyroid": "np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)",
    "Schwarz Diamond": "np.sin(X)*np.sin(Y)*np.sin(Z) + np.sin(X)*np.cos(Y)*np.cos(Z) + np.cos(X)*np.sin(Y)*np.cos(Z) + np.cos(X)*np.cos(Y)*np.sin(Z)",
    "Schwarz Primitive": "np.cos(X) + np.cos(Y) + np.cos(Z)",
    "Neovius": "3*(np.cos(X)+np.cos(Y)+np.cos(Z)) + 4*np.cos(X)*np.cos(Y)*np.cos(Z)",
    "Lidinoid": "np.sin(2*X)*np.cos(Y)*np.sin(Z) + np.sin(2*Y)*np.cos(Z)*np.sin(X) + np.sin(2*Z)*np.cos(X)*np.sin(Y) - np.cos(2*X)*np.cos(2*Y) - np.cos(2*Y)*np.cos(2*Z) - np.cos(2*Z)*np.cos(2*X)",
    "Schoen I-WP": "2*(np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z) + np.cos(Z)*np.cos(X)) - (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z))",
    "Schoen F-RD": "4*np.cos(X)*np.cos(Y)*np.cos(Z) - np.cos(2*X)*np.cos(2*Y) - np.cos(2*Y)*np.cos(2*Z) - np.cos(2*Z)*np.cos(2*X)",
    "Fischer-Koch S": "np.cos(2*X)*np.sin(Y)*np.cos(Z) + np.cos(2*Y)*np.sin(Z)*np.cos(X) + np.cos(2*Z)*np.sin(X)*np.cos(Y)",
    "Cross Layers": "np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z)",
    "Tubular Matrix": "10 - (np.sin(X)**2 + np.sin(Y)**2 + np.sin(Z)**2)"
}


def evaluate_lattice_string(eq, X, Y, Z):
    try:
        return eval(eq, {"__builtins__": {}}, {"np": np, "X": X, "Y": Y, "Z": Z})
    except Exception:
        return np.zeros_like(X)


def normalize(f):
    f = f - np.mean(f)
    return f / (np.max(np.abs(f)) + 1e-8)


def generate_multi_field_lattice(
    resolution=64,
    periods=4.0,

    base_style="Gyroid",
    pressure_style="Schwarz Diamond",

    density_base=0.5,
    density_max=0.8,

    combined_pressure_field=None
):

    # -----------------------------
    # DOMAIN
    # -----------------------------
    coords = np.linspace(-periods*np.pi, periods*np.pi, resolution)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # -----------------------------
    # FIELD EVAL
    # -----------------------------
    base_expr = LATTICE_LIBRARY[base_style]
    press_expr = LATTICE_LIBRARY[pressure_style]

    f_base = evaluate_lattice_string(base_expr, X, Y, Z)
    f_press = evaluate_lattice_string(press_expr, X, Y, Z)

    # -----------------------------
    # NORMALIZE TPMS FIELDS
    # -----------------------------
    f_base = normalize(f_base)
    f_press = normalize(f_press)

    # -----------------------------
    # PRESSURE BLEND (ONLY SHAPE)
    # -----------------------------
    W = np.zeros_like(f_base) if combined_pressure_field is None else np.clip(combined_pressure_field, 0.0, 1.0)

    field = (1.0 - W) * f_base + W * f_press

    # -----------------------------
    # FIXED DENSITY MODEL
    # -----------------------------
    density_base = np.clip(density_base, 0.0, 1.0)

    iso = -1.0 + 2.0 * density_base

    # -----------------------------
    # FINAL FIELD
    # -----------------------------
    final_field = field - iso

    # -----------------------------
    # BOUNDARIES
    # -----------------------------
    final_field[[0, -1], :, :] = 1
    final_field[:, [0, -1], :] = 1
    final_field[:, :, [0, -1]] = 1

    # -----------------------------
    # MARCHING CUBES
    # -----------------------------
    verts, faces, _, _ = measure.marching_cubes(final_field, level=0.0)

    # -----------------------------
    # NORMALIZE SPACE
    # -----------------------------
    spacing = 2.0 / (resolution - 1)
    verts = (verts * spacing) - 1.0

    # -----------------------------
    # BUILD MESH
    # -----------------------------
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        return mesh
    except Exception as e:
        print("Mesh error:", e)
        return None