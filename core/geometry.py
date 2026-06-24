import numpy as np
import trimesh

def load_and_voxelize_mesh(file_path, resolution=64):
    mesh = trimesh.load(file_path)
    assert isinstance(mesh, trimesh.Trimesh), "Loaded mesh is not a valid Trimesh object." 
    
    if not mesh.is_watertight:
        mesh.fill_holes()

    
    centroid = mesh.centroid
    extents = mesh.extents
    max_extent = np.max(extents)
    
    mesh.vertices = (mesh.vertices - centroid) * (1.8 / max_extent)
    
    voxel_grid = mesh.voxelized(pitch=2.0/resolution)
    cad_mask = voxel_grid.matrix.astype(bool)
    
    if cad_mask.shape != (resolution, resolution, resolution):
        padded_mask = np.zeros((resolution, resolution, resolution), dtype=bool)
        off = (np.array(cad_mask.shape) - resolution) // 2
        target_shape = np.array(cad_mask.shape)
        padded_mask[:target_shape[0], :target_shape[1], :target_shape[2]] = cad_mask
        cad_mask = padded_mask

    return mesh, cad_mask