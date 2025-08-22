import numpy as np


def optimal_controls_from_gradient(grad_vx: np.ndarray, grad_vy: np.ndarray, a_p_max: float, a_e_max: float, r_x: float = None, r_y: float = None, bias_evader_away_from_r: float = 0.0):
	# Base direction from gradient w.r.t. velocity components
	norm_v = np.sqrt(grad_vx * grad_vx + grad_vy * grad_vy) + 1e-12
	dir_vx = grad_vx / norm_v
	dir_vy = grad_vy / norm_v
	# Optional bias: push evader outward along relative position direction r-hat
	if r_x is not None and r_y is not None and bias_evader_away_from_r > 0.0:
		norm_r = np.sqrt(r_x * r_x + r_y * r_y) + 1e-12
		dir_rx = r_x / norm_r
		dir_ry = r_y / norm_r
		mix_x = (1.0 - bias_evader_away_from_r) * dir_vx + bias_evader_away_from_r * dir_rx
		mix_y = (1.0 - bias_evader_away_from_r) * dir_vy + bias_evader_away_from_r * dir_ry
		nrm = np.sqrt(mix_x * mix_x + mix_y * mix_y) + 1e-12
		dir_vx = mix_x / nrm
		dir_vy = mix_y / nrm
	# Evader maximizes value -> align with direction; pursuer minimizes -> oppose gradient direction only
	a_e = a_e_max * np.stack([dir_vx, dir_vy], axis=-1)
	a_p = -a_p_max * np.stack([dir_vx, dir_vy], axis=-1)
	return a_p, a_e