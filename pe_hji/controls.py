import numpy as np


def optimal_controls_from_gradient(grad_vx: np.ndarray, grad_vy: np.ndarray, a_p_max: float, a_e_max: float):
	"""
	Extract feedback accelerations from the value-function gradient w.r.t. v.

	Isaacs term for accelerations is p_v·(a_e - a_p). The maximizing evader chooses
	a_e parallel to p_v with magnitude a_e_max, while the minimizing pursuer should 
	oppose this term. In our relative dynamics convention, minimizing p_v·(a_e - a_p)
	results in the pursuer choosing a_p parallel to -p_v (we implement that). The
	evader aligns with +p_v.
	"""
	# Direction from gradient w.r.t. velocity components
	norm_v = np.sqrt(grad_vx * grad_vx + grad_vy * grad_vy) + 1e-12
	dir_vx = grad_vx / norm_v
	dir_vy = grad_vy / norm_v
	# Evader aligns with gradient (maximizes V); pursuer opposes (minimizes V)
	a_e = a_e_max * np.stack([dir_vx, dir_vy], axis=-1)
	a_p = -a_p_max * np.stack([dir_vx, dir_vy], axis=-1)
	return a_p, a_e