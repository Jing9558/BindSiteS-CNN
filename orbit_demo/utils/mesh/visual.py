"""
orbit.utils.mesh.visual

"""
import numpy as np
import trimesh


def interpolate(values, cmap, low=-1, high=1, dtype=np.uint8):
    if cmap is None:
        cmap = trimesh.visual.linear_color_map
    else:
        from matplotlib.pyplot import get_cmap
        cmap = get_cmap(cmap)
    values = np.asanyarray(values, dtype=float).ravel()
    if low is None or high is None:
        colors = cmap((values - values.min()) / values.ptp())
    else:
        colors = cmap((values - low) / (high - low))
    rgba = trimesh.visual.to_rgba(colors, dtype=dtype)
    return rgba
