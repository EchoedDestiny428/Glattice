import numpy as np


def calculate_multi_pressure_field(
    resolution=64,
    click_list=None,
    influence_radius=0.3
):

    field = np.zeros(
        (resolution, resolution, resolution),
        dtype=np.float32
    )

    if not click_list:
        return field

    coords = np.linspace(-1, 1, resolution)

    X, Y, Z = np.meshgrid(
        coords,
        coords,
        coords,
        indexing="ij"
    )

    for cx, cy, cz in click_list:

        d2 = (
            (X - cx) ** 2 +
            (Y - cy) ** 2 +
            (Z - cz) ** 2
        )

        local = np.exp(
            -d2 / (2 * influence_radius**2)
        )

        field = np.maximum(field, local)

    return field