import numpy as np


def optimal_controls_from_gradient(grad_vx: np.ndarray, grad_vy: np.ndarray, a_p_max: float, a_e_max: float):
	# Controls depend only on gradient w.r.t. velocity components
	norm = np.sqrt(grad_vx * grad_vx + grad_vy * grad_vy) + 1e-12
	dirx = grad_vx / norm
	diry = grad_vy / norm
	# Evader maximizes value -> align with gradient; pursuer minimizes -> oppose gradient
	a_e = a_e_max * np.stack([dirx, diry], axis=-1)
	a_p = -a_p_max * np.stack([dirx, diry], axis=-1)
	return a_p, a_e