import numpy as np
import trimesh


def load_and_voxelize_mesh(file_path, resolution=64):

    loaded = trimesh.load(file_path)

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]

        if not meshes:
            raise ValueError("OBJ contains no mesh geometry")

        mesh = trimesh.util.concatenate(meshes)

    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    else:
        raise ValueError("Unsupported mesh type")

    mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight:
        mesh.fill_holes()

    center = mesh.bounding_box.centroid

    extents = mesh.bounding_box.extents
    max_extent = np.max(extents)

    if max_extent <= 0:
        raise ValueError("Mesh has zero dimensions")

    scale = 1.8 / max_extent

    mesh.vertices = (mesh.vertices - center) * scale

    voxel_grid = mesh.voxelized(pitch=2.0 / resolution)

    cad_mask = voxel_grid.matrix.astype(bool)

    target = np.zeros(
        (resolution, resolution, resolution),
        dtype=bool
    )

    src_shape = np.array(cad_mask.shape)

    copy_shape = np.minimum(
        src_shape,
        np.array([resolution, resolution, resolution])
    )

    target[
        :copy_shape[0],
        :copy_shape[1],
        :copy_shape[2]
    ] = cad_mask[
        :copy_shape[0],
        :copy_shape[1],
        :copy_shape[2]
    ]

    return mesh, target