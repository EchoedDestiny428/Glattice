import numpy as np
import trimesh

from skimage import measure


LATTICE_LIBRARY = {
    "Gyroid":
        "np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)",

    "Schwarz Diamond":
        "np.sin(X)*np.sin(Y)*np.sin(Z) + "
        "np.sin(X)*np.cos(Y)*np.cos(Z) + "
        "np.cos(X)*np.sin(Y)*np.cos(Z) + "
        "np.cos(X)*np.cos(Y)*np.sin(Z)",

    "Schwarz Primitive":
        "np.cos(X) + np.cos(Y) + np.cos(Z)",

    "Neovius":
        "3*(np.cos(X)+np.cos(Y)+np.cos(Z)) + "
        "4*np.cos(X)*np.cos(Y)*np.cos(Z)",

    "Lidinoid":
        "np.sin(2*X)*np.cos(Y)*np.sin(Z) + "
        "np.sin(2*Y)*np.cos(Z)*np.sin(X) + "
        "np.sin(2*Z)*np.cos(X)*np.sin(Y) - "
        "np.cos(2*X)*np.cos(2*Y) - "
        "np.cos(2*Y)*np.cos(2*Z) - "
        "np.cos(2*Z)*np.cos(2*X)",

    "Schoen I-WP":
        "2*(np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z) + np.cos(Z)*np.cos(X)) "
        "- (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z))",

    "Schoen F-RD":
        "4*np.cos(X)*np.cos(Y)*np.cos(Z) "
        "- np.cos(2*X)*np.cos(2*Y) "
        "- np.cos(2*Y)*np.cos(2*Z) "
        "- np.cos(2*Z)*np.cos(2*X)",

    "Fischer-Koch S":
        "np.cos(2*X)*np.sin(Y)*np.cos(Z) + "
        "np.cos(2*Y)*np.sin(Z)*np.cos(X) + "
        "np.cos(2*Z)*np.sin(X)*np.cos(Y)",

    "Cross Layers":
        "np.cos(X)*np.cos(Y) + np.cos(Y)*np.cos(Z)",

    "Tubular Matrix":
        "10 - (np.sin(X)**2 + np.sin(Y)**2 + np.sin(Z)**2)"
}


def evaluate_lattice_string(
    equation_str,
    X,
    Y,
    Z
):
    """
    Evaluate a TPMS equation safely.
    """

    try:

        allowed_locals = {
            "np": np,
            "X": X,
            "Y": Y,
            "Z": Z
        }

        return eval(
            equation_str,
            {"__builtins__": {}},
            allowed_locals
        )

    except Exception as e:

        print(
            f"Equation evaluation failed: {e}"
        )

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
    """
    Generate a pressure-adaptive TPMS lattice.

    Entire pipeline operates in normalized
    coordinate space [-1,1].
    """

    coords = np.linspace(
        -periods * np.pi,
        periods * np.pi,
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
        else LATTICE_LIBRARY.get(
            base_style,
            LATTICE_LIBRARY["Gyroid"]
        )
    )

    pressure_expr = (
        custom_press_eq
        if pressure_style == "Custom Equation"
        else LATTICE_LIBRARY.get(
            pressure_style,
            LATTICE_LIBRARY["Schwarz Diamond"]
        )
    )

    f_base = evaluate_lattice_string(
        base_expr,
        X,
        Y,
        Z
    )

    f_pressure = evaluate_lattice_string(
        pressure_expr,
        X,
        Y,
        Z
    )

    # ---------------------------------
    # Topology blending
    # ---------------------------------

    if combined_pressure_field is None:

        final_field = (
            f_base -
            base_thickness
        )

    else:

        W = np.clip(
            combined_pressure_field,
            0.0,
            1.0
        )

        blended_tpms = (
            (1.0 - W) * f_base +
            W * f_pressure
        )

        threshold = (
            base_thickness +
            W * (
                max_pressure_thickness -
                base_thickness
            )
        )

        final_field = (
            blended_tpms -
            threshold
        )

    # ---------------------------------
    # Force closed volume boundaries
    # ---------------------------------

    final_field[0, :, :] = 1.0
    final_field[-1, :, :] = 1.0

    final_field[:, 0, :] = 1.0
    final_field[:, -1, :] = 1.0

    final_field[:, :, 0] = 1.0
    final_field[:, :, -1] = 1.0

    try:

        verts, faces, normals, values = (
            measure.marching_cubes(
                final_field,
                level=0
            )
        )

    except Exception as e:

        print(
            f"Marching cubes failed: {e}"
        )

        return None

    # ---------------------------------
    # Convert voxel coordinates
    # into normalized CAD space [-1,1]
    # ---------------------------------

    spacing = (
        2.0 /
        (resolution - 1)
    )

    verts = (
        verts * spacing
    ) - 1.0

    try:

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False
        )

        mesh.merge_vertices()

        mesh.remove_unreferenced_vertices()

        try:
            mesh.process(
                validate=True
            )
        except Exception:
            pass

        try:
            mesh.fill_holes()
        except Exception:
            pass

        return mesh

    except Exception as e:

        print(
            f"Mesh construction failed: {e}"
        )

        return None