import numpy as np
import trimesh


def _load_mesh(file_path):
    """
    Robust mesh loader supporting STL and OBJ.
    Returns a single trimesh.Trimesh.
    """

    loaded = trimesh.load(file_path)

    if isinstance(loaded, trimesh.Scene):

        meshes = [
            g
            for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]

        if not meshes:
            raise ValueError(
                "No valid mesh geometry found."
            )

        mesh = trimesh.util.concatenate(meshes)

    elif isinstance(loaded, trimesh.Trimesh):

        mesh = loaded

    else:

        raise ValueError(
            f"Unsupported mesh type: {type(loaded)}"
        )

    return mesh


def _normalize_mesh(mesh):
    """
    Normalize mesh into a centered [-1,1] domain.
    Returns normalized mesh and metadata.
    """

    mesh = mesh.copy()

    bounds_min = mesh.vertices.min(axis=0)
    bounds_max = mesh.vertices.max(axis=0)

    center = (
        bounds_min + bounds_max
    ) / 2.0

    extents = (
        bounds_max - bounds_min
    )

    max_extent = np.max(extents)

    if max_extent <= 0:
        raise ValueError(
            "Mesh has zero size."
        )

    scale = 2.0 / max_extent

    mesh.vertices = (
        mesh.vertices - center
    ) * scale

    metadata = {
        "center": center,
        "scale": scale,
        "extents": extents
    }

    return mesh, metadata


def _voxelize_mesh(
    mesh,
    resolution
):
    """
    Convert normalized mesh into voxel occupancy.
    Always returns a fixed resolution cube.
    """

    pitch = 2.0 / resolution

    voxel_grid = mesh.voxelized(
        pitch=pitch
    )

    cad_mask = voxel_grid.matrix.astype(
        bool
    )

    target = np.zeros(
        (
            resolution,
            resolution,
            resolution
        ),
        dtype=bool
    )

    source_shape = np.array(
        cad_mask.shape
    )

    target_shape = np.array(
        target.shape
    )

    copy_shape = np.minimum(
        source_shape,
        target_shape
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

    return target


def load_and_voxelize_mesh(
    file_path,
    resolution=64
):
    """
    Main geometry pipeline.

    Returns
    -------
    mesh : trimesh.Trimesh
        Normalized mesh in [-1,1]

    cad_mask : np.ndarray
        Boolean occupancy grid

    metadata : dict
        Normalization information
    """

    mesh = _load_mesh(file_path)

    mesh.remove_unreferenced_vertices()

    try:
        mesh.process(validate=True)
    except Exception:
        pass

    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
        except Exception:
            pass

    mesh, metadata = _normalize_mesh(
        mesh
    )

    cad_mask = _voxelize_mesh(
        mesh,
        resolution
    )

    return (
        mesh,
        cad_mask,
        metadata
    )