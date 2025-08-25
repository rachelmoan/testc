import numpy as np
from typing import Tuple


def relative_dynamics(state: np.ndarray, a_p: np.ndarray, a_e: np.ndarray) -> np.ndarray:
	# state = [rx, ry, rvx, rvy]
	drx = state[2]
	dry = state[3]
	drvx = a_e[0] - a_p[0]
	drvy = a_e[1] - a_p[1]
	return np.array([drx, dry, drvx, drvy], dtype=float)


def stackelberg_hamiltonian(rvx: np.ndarray, rvy: np.ndarray, px: np.ndarray, py: np.ndarray, pvx: np.ndarray, pvy: np.ndarray, a_p_max: float, a_e_max: float) -> np.ndarray:
	# H(x,p) = p_r · v_rel + (a_e_max - a_p_max) * ||p_v||
	return px * rvx + py * rvy + (a_e_max - a_p_max) * np.sqrt(pvx * pvx + pvy * pvy + 1e-16)


def bounds_hamiltonian_partials(vx_axis: np.ndarray, vy_axis: np.ndarray, a_p_max: float, a_e_max: float) -> Tuple[float, float, float, float]:
	alpha1 = float(np.max(np.abs(vx_axis)))
	alpha2 = float(np.max(np.abs(vy_axis)))
	alpha3 = abs(a_e_max - a_p_max)
	alpha4 = abs(a_e_max - a_p_max)
	return alpha1, alpha2, alpha3, alpha4


# 3D versions
def stackelberg_hamiltonian_3d(rvx: np.ndarray, rvy: np.ndarray, rvz: np.ndarray, px: np.ndarray, py: np.ndarray, pz: np.ndarray, pvx: np.ndarray, pvy: np.ndarray, pvz: np.ndarray, a_p_max: float, a_e_max: float) -> np.ndarray:
	# H(x,p) = p_r · v_rel + (a_e_max - a_p_max) * ||p_v||
	return (
		px * rvx + py * rvy + pz * rvz
		+ (a_e_max - a_p_max) * np.sqrt(pvx * pvx + pvy * pvy + pvz * pvz + 1e-16)
	)


def bounds_hamiltonian_partials_3d(vx_axis: np.ndarray, vy_axis: np.ndarray, vz_axis: np.ndarray, a_p_max: float, a_e_max: float) -> Tuple[float, float, float, float, float, float]:
	alpha1 = float(np.max(np.abs(vx_axis)))
	alpha2 = float(np.max(np.abs(vy_axis)))
	alpha3 = float(np.max(np.abs(vz_axis)))
	alpha4 = abs(a_e_max - a_p_max)
	alpha5 = abs(a_e_max - a_p_max)
	alpha6 = abs(a_e_max - a_p_max)
	return alpha1, alpha2, alpha3, alpha4, alpha5, alpha6