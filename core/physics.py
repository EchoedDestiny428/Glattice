import numpy as np


def calculate_multi_pressure_field(
    resolution=64,
    click_list=None,
    influence_radius=0.35,
):
    field = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    coords = np.linspace(-1.0, 1.0, resolution)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    radius_sq = influence_radius ** 2

    # -------------------------------------------------
    # BASE FIELD
    # -------------------------------------------------
    base_field = np.ones_like(field) * 0.15
    field = np.maximum(field, base_field)

    # -------------------------------------------------
    # CLICK CONTRIBUTIONS
    # -------------------------------------------------
    if click_list:

        for point in click_list:

            cx, cy, cz = point

            dist_sq = (
                (X - cx) ** 2 +
                (Y - cy) ** 2 +
                (Z - cz) ** 2
            )

            pressure = np.exp(-dist_sq / (2.0 * radius_sq))

            field = np.maximum(field, pressure)

    # -------------------------------------------------
    # NORMALIZE
    # -------------------------------------------------
    field = field / (np.max(field) + 1e-8)

    return field