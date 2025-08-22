import numpy as np


def optimal_controls_from_gradient(grad_vx: np.ndarray, grad_vy: np.ndarray, a_p_max: float, a_e_max: float):
	# Base direction from gradient w.r.t. velocity components
	norm_v = np.sqrt(grad_vx * grad_vx + grad_vy * grad_vy) + 1e-12
	dir_vx = grad_vx / norm_v
	dir_vy = grad_vy / norm_v
	# Evader maximizes value -> align with direction; pursuer minimizes -> oppose gradient direction
	a_e = a_e_max * np.stack([dir_vx, dir_vy], axis=-1)
	a_p = -a_p_max * np.stack([dir_vx, dir_vy], axis=-1)
	return a_p, a_e