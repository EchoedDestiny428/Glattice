import numpy as np

def calculate_multi_pressure_field(resolution=64, click_list=None, influence_radius=0.3):
    """Combines multiple 3D click coordinates into a unified pressure field grid.

    Args:
        resolution (int): Dimension size of the 3D grid space.
        click_list (list): Collection of (x, y, z) tuples representing click coordinates.
        influence_radius (float): Radius of effect (sigma) for the pressure falloff.

    Returns:
        numpy.ndarray: Combined 3D matrix containing field values ranging from 0.0 to 1.0.
    """
    master_field = np.zeros((resolution, resolution, resolution))
    
    if not click_list:
        return master_field
        
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    for click in click_list:
        cx, cy, cz = click
        distance = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        
        local_field = np.exp(- (distance**2) / (2 * (influence_radius**2)))
        master_field = np.maximum(master_field, local_field)
        
    return master_field