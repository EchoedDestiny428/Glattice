import numpy as np
import trimesh
from trimesh import Trimesh

def load_and_voxelize_mesh(file_path, resolution=64):
    """Loads a 3D CAD mesh file, fits it inside normalized bounds, and creates a 3D binary mask of its interior space.

    Args:
        file_path (str): System path location to the imported STL or OBJ file.
        resolution (int): Target grid size matching the lattice space.

    Returns:
        tuple: (trimesh.Trimesh object of scaled mesh, 3D numpy boolean mask grid of interior points).
    """
    mesh = trimesh.load(file_path)
    assert isinstance(mesh, trimesh.Trimesh), "Loaded mesh is not a valid Trimesh object."
    
    bounds = mesh.bounds
    max_side = np.max(bounds[1] - bounds[0])
    center = mesh.centroid
    mesh.vertices = (mesh.vertices - center) * (1.8 / max_side)
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    
    inside_points = mesh.contains(points)
    cad_mask = inside_points.reshape((resolution, resolution, resolution))
    
    return mesh, cad_mask