# This python file consists of all the functions necessary for the OSOM data depth conversion for easier comparisons.
import numpy as np


# for calculation of depth
def z(hc, s_rho, Cs_r, h, zeta):
    return zeta + (zeta + h) * ((hc * s_rho + h * Cs_r) / (hc + h))


# for observation depth conversion
def obs_to_z(d, zeta, anchor=False, h_obs=None):
    if anchor:
        z = -h_obs + d
    else:
        z = zeta - d
    return z


# for interpolation
def interp(model_z, model_values, obs_z):
    """
    interpolate one model profile to observation z coords

    model_z: model profile in z coordinates
    model_values: model temperature or salinity at corresponding positions
    obs_z: observation in z

    """
    model_z = np.asarray(model_z, dtype=float)
    model_values = np.asanyarray(model_values, dtype=float)

    # making sure that all the values are valid
    if not (np.isfinite(model_z).all() and np.isfinite(model_values).all()):
        raise ValueError("data contain missing or infinite values")

    order = np.argsort(model_z)
    model_z = model_z[order]
    model_values = model_values[order]

    model_interp_z = np.interp(obs_z, model_z, model_values)
    return model_interp_z


# uncertainty calculation
def z_error(model_z_w, obs_z):
    """carry half the local layer thickness as the vertical interpolation uncertainty."""
    zw = np.asarray(model_z_w, dtype=float)
    obs_z = np.asarray(obs_z, dtype=float)

    thickness = np.diff(zw)
    # determining which index the obs_z lies in zw
    k = np.searchsorted(zw, obs_z, side="right") - 1
    # confirming that the index is nonnegative and does not exceed the most right bound
    k = np.clip(k, 0, len(thickness) - 1)

    # one more criteria to confirm that obs_z lies in between zw, which should be the case
    # since zw[0] is the bottom and zw[-1] is the surface, obs_z must lie within the two
    inside = (obs_z >= zw[0]) & (obs_z <= zw[-1])

    # if criteria not met nan is returned
    return np.where(inside, thickness[k] / 2, np.nan)
