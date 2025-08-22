import numpy as np
from typing import Tuple


def relative_dynamics(state: np.ndarray, a_p: np.ndarray, a_e: np.ndarray) -> np.ndarray:
	"""
	Relative dynamics for two 2D double-integrator point masses.

	State x = [r_x, r_y, r_vx, r_vy] where
	- r = e_pos - p_pos
	- v_rel = e_vel - p_vel
	Controls are accelerations a_p (pursuer) and a_e (evader).

	Equations
	- ṙ = v_rel
	- v̇_rel = a_e - a_p
	"""
	# state = [rx, ry, rvx, rvy]
	drx = state[2]
	dry = state[3]
	drvx = a_e[0] - a_p[0]
	drvy = a_e[1] - a_p[1]
	return np.array([drx, dry, drvx, drvy], dtype=float)


def stackelberg_hamiltonian(rvx: np.ndarray, rvy: np.ndarray, px: np.ndarray, py: np.ndarray, pvx: np.ndarray, pvy: np.ndarray, a_p_max: float, a_e_max: float) -> np.ndarray:
	"""
	Isaacs/Stackelberg Hamiltonian for the relative dynamics and capture target.

	For value function V(x,t), with cost based on capture, the Hamiltonian term is
	H(x,p) = p_r · v_rel + max_{||a_e||≤a_e_max} min_{||a_p||≤a_p_max} p_v · (a_e - a_p)
	       = p_r · v_rel + (a_e_max - a_p_max) ||p_v||,  where p_v = ∂V/∂v

	We evaluate H at central gradients p = ∂V and add LF dissipation outside.
	"""
	# H(x,p) = p_r · v_rel + (a_e_max - a_p_max) * ||p_v||
	return px * rvx + py * rvy + (a_e_max - a_p_max) * np.sqrt(pvx * pvx + pvy * pvy + 1e-16)


def bounds_hamiltonian_partials(vx_axis: np.ndarray, vy_axis: np.ndarray, a_p_max: float, a_e_max: float) -> Tuple[float, float, float, float]:
	"""
	Compute Lax–Friedrichs artificial viscosity (alpha) bounds per dimension.

	- For r components, |∂H/∂p_r| ≤ |v_rel|, so use max |r_vx|, |r_vy| over grid axes.
	- For v components, |∂H/∂p_v| ≤ |a_e_max - a_p_max|.
	"""
	alpha1 = float(np.max(np.abs(vx_axis)))
	alpha2 = float(np.max(np.abs(vy_axis)))
	alpha3 = abs(a_e_max - a_p_max)
	alpha4 = abs(a_e_max - a_p_max)
	return alpha1, alpha2, alpha3, alpha4