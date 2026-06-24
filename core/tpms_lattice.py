import numpy as np
from skimage import measure
import trimesh

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

def evaluate_lattice_string(equation_str, X, Y, Z):
    """Evaluates a raw string formula using the provided coordinate grids.

    Args:
        equation_str (str): The mathematical equation text.
        X, Y, Z (numpy.ndarray): 3D grids representing the coordinate system.

    Returns:
        numpy.ndarray: Evaluated scalar values across the 3D grid space.
    """
    try:
        allowed_locals = {"np": np, "X": X, "Y": Y, "Z": Z}
        return eval(equation_str, {"__builtins__": None}, allowed_locals)
    except Exception as e:
        print(f"Error parsing custom equation: {e}")
        return np.zeros_like(X)

def generate_multi_field_lattice(resolution=64, periods=3.0, 
                                 base_style="Gyroid", custom_base_eq="",
                                 pressure_style="Schwarz Diamond", custom_press_eq="",
                                 base_thickness=0.0, max_pressure_thickness=0.4,
                                 combined_pressure_field=None):
    """Generates a hybrid 3D mesh by blending two distinct TPMS lattice structures.

    Args:
        resolution (int): Dimension size of the 3D grid.
        periods (float): Frequency multiplier for the cell repetition.
        base_style (str): Key name for the default background lattice.
        custom_base_eq (str): Raw math formula if base_style is 'Custom Equation'.
        pressure_style (str): Key name for the lattice inside high-pressure areas.
        custom_press_eq (str): Raw math formula if pressure_style is 'Custom Equation'.
        base_thickness (float): Baseline thickness parameter for the walls.
        max_pressure_thickness (float): Maximum wall thickness target at peak pressure locations.
        combined_pressure_field (numpy.ndarray): 3D weight grid (0.0 to 1.0) for morphing transitions.

    Returns:
        trimesh.Trimesh: Consolidated 3D surface model, or None if evaluation fails.
    """
    scale = np.pi * periods
    x = np.linspace(-scale, scale, resolution)
    y = np.linspace(-scale, scale, resolution)
    z = np.linspace(-scale, scale, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    base_expr = custom_base_eq if base_style == "Custom Equation" else LATTICE_LIBRARY.get(base_style, LATTICE_LIBRARY["Gyroid"])
    press_expr = custom_press_eq if pressure_style == "Custom Equation" else LATTICE_LIBRARY.get(pressure_style, LATTICE_LIBRARY["Schwarz Diamond"])
    
    f_base = evaluate_lattice_string(base_expr, X, Y, Z)
    f_pressure = evaluate_lattice_string(press_expr, X, Y, Z)
    
    if combined_pressure_field is not None:
        W = combined_pressure_field
        final_field = (1.0 - W) * f_base + W * f_pressure
        density_boost = max_pressure_thickness - base_thickness
        threshold_grid = base_thickness + (W * density_boost)
    else:
        final_field = f_base
        threshold_grid = np.full_like(final_field, base_thickness)
        
    solid_mask = np.abs(final_field) - (0.3 + threshold_grid)
    
    try:
        verts, faces, normals, values = measure.marching_cubes(solid_mask, level=0)
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except ValueError:
        return None