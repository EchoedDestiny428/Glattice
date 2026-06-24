import numpy as np
import trimesh

from skimage import measure


LATTICE_LIBRARY = {
    "Gyroid":
        "np.sin(X)*np.cos(Y)+np.sin(Y)*np.cos(Z)+np.sin(Z)*np.cos(X)",

    "Schwarz Diamond":
        "np.sin(X)*np.sin(Y)*np.sin(Z)+"
        "np.sin(X)*np.cos(Y)*np.cos(Z)+"
        "np.cos(X)*np.sin(Y)*np.cos(Z)+"
        "np.cos(X)*np.cos(Y)*np.sin(Z)",

    "Schwarz Primitive":
        "np.cos(X)+np.cos(Y)+np.cos(Z)",

    "Neovius":
        "3*(np.cos(X)+np.cos(Y)+np.cos(Z))+"
        "4*np.cos(X)*np.cos(Y)*np.cos(Z)",

    "Lidinoid":
        "np.sin(2*X)*np.cos(Y)*np.sin(Z)+"
        "np.sin(2*Y)*np.cos(Z)*np.sin(X)+"
        "np.sin(2*Z)*np.cos(X)*np.sin(Y)-"
        "np.cos(2*X)*np.cos(2*Y)-"
        "np.cos(2*Y)*np.cos(2*Z)-"
        "np.cos(2*Z)*np.cos(2*X)"
}


def evaluate_lattice_string(expr, X, Y, Z):

    allowed = {
        "np": np,
        "X": X,
        "Y": Y,
        "Z": Z
    }

    try:
        return eval(
            expr,
            {"__builtins__": {}},
            allowed
        )

    except Exception as e:
        print(e)
        return np.zeros_like(X)


def generate_multi_field_lattice(
    resolution=64,
    periods=4.0,
    base_style="Gyroid",
    custom_base_eq="",
    pressure_style="Schwarz Diamond",
    custom_press_eq="",
    base_thickness=0.0,
    max_pressure_thickness=0.3,
    combined_pressure_field=None
):

    scale = np.pi * periods

    coords = np.linspace(
        -scale,
        scale,
        resolution
    )

    X, Y, Z = np.meshgrid(
        coords,
        coords,
        coords,
        indexing="ij"
    )

    base_expr = (
        custom_base_eq
        if base_style == "Custom Equation"
        else LATTICE_LIBRARY[base_style]
    )

    pressure_expr = (
        custom_press_eq
        if pressure_style == "Custom Equation"
        else LATTICE_LIBRARY[pressure_style]
    )

    f_base = evaluate_lattice_string(
        base_expr,
        X,
        Y,
        Z
    )

    f_press = evaluate_lattice_string(
        pressure_expr,
        X,
        Y,
        Z
    )

    if combined_pressure_field is None:

        final_field = f_base - base_thickness

    else:

        W = np.clip(
            combined_pressure_field,
            0,
            1
        )

        blended = (
            (1.0 - W) * f_base +
            W * f_press
        )

        threshold = (
            base_thickness +
            W *
            (
                max_pressure_thickness -
                base_thickness
            )
        )

        final_field = blended - threshold

    final_field[0, :, :] = 1
    final_field[-1, :, :] = 1

    final_field[:, 0, :] = 1
    final_field[:, -1, :] = 1

    final_field[:, :, 0] = 1
    final_field[:, :, -1] = 1

    try:

        verts, faces, normals, values = (
            measure.marching_cubes(
                final_field,
                level=0
            )
        )

        spacing = (
            2.0 * scale
        ) / (
            resolution - 1
        )

        verts = (
            verts * spacing
        ) - scale

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
        )

        mesh.merge_vertices()

        mesh.remove_unreferenced_vertices()

        mesh.fill_holes()

        mesh.fix_normals()

        return mesh

    except Exception as e:

        print(
            f"Lattice generation failed: {e}"
        )

        return None