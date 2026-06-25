import numpy as np


def calculate_multi_pressure_field(
    resolution=64,
    click_list=None,
    influence_radius=0.35,
):
    """
    Generate a normalized pressure field.

    Parameters
    ----------
    resolution : int
        Grid resolution.

    click_list : list
        List of pressure points in normalized
        coordinates [-1,1].

    influence_radius : float
        Gaussian radius in normalized units.

    Returns
    -------
    np.ndarray
        Pressure field in range [0,1].
    """

    field = np.zeros(
        (
            resolution,
            resolution,
            resolution
        ),
        dtype=np.float32
    )

    if not click_list:
        return field

    coords = np.linspace(
        -1.0,
        1.0,
        resolution
    )

    X, Y, Z = np.meshgrid(
        coords,
        coords,
        coords,
        indexing="ij"
    )

    radius_sq = influence_radius ** 2

    for point in click_list:

        cx, cy, cz = point

        dist_sq = (
            (X - cx) ** 2 +
            (Y - cy) ** 2 +
            (Z - cz) ** 2
        )

        pressure = np.exp(
            -dist_sq /
            (2.0 * radius_sq)
        )

        field = np.maximum(
            field,
            pressure
        )

    return field