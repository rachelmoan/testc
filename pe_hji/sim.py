import numpy as np
from typing import Tuple
from .grid import Grid4D
from .controls import optimal_controls_from_gradient


def _interp_central_gradients(V: np.ndarray, grid: Grid4D, state: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
	rx, ry, rvx, rvy = grid.axes
	idxs = []
	ws = []
	coords = [state[0], state[1], state[2], state[3]]
	for axis_vals, val in zip((rx, ry, rvx, rvy), coords):
		i = int(np.clip(np.searchsorted(axis_vals, val) - 1, 0, len(axis_vals) - 2))
		idxs.append(i)
		w = (val - axis_vals[i]) / (axis_vals[i+1] - axis_vals[i])
		ws.append(w)

	# Trilinear-like 4D interpolation of V and gradients via finite differences
	# Compute local central differences using neighbours along each axis
	# Gather 2^4 corner values
	corners = np.zeros(16)
	for b in range(16):
		ix = idxs[0] + ((b >> 0) & 1)
		iy = idxs[1] + ((b >> 1) & 1)
		ivx = idxs[2] + ((b >> 2) & 1)
		ivy = idxs[3] + ((b >> 3) & 1)
		corners[b] = V[ix, iy, ivx, ivy]

	# Interpolate V
	wx, wy, wvx, wvy = ws
	V_interp = 0.0
	for b in range(16):
		cx = wx if (b & 1) else (1 - wx)
		cy = wy if (b & 2) else (1 - wy)
		cvx = wvx if (b & 4) else (1 - wvx)
		cvy = wvy if (b & 8) else (1 - wvy)
		V_interp += corners[b] * cx * cy * cvx * cvy

	# Approximate gradient wrt velocity components by central difference of interpolated V values along v-axes
	dvx = grid.dr[2]
	dvy = grid.dr[3]
	# Shift a small epsilon in index space
	def shift_and_interp(axis_idx: int, sign: int):
		coords_shift = coords.copy()
		coords_shift[axis_idx] += sign * (grid.dr[axis_idx] * 0.5)
		coords_shift[axis_idx] = float(np.clip(coords_shift[axis_idx], grid.axes[axis_idx][0], grid.axes[axis_idx][-1]))
		return _interp_scalar(V, grid, coords_shift)

	Vp_vx_plus = shift_and_interp(2, +1)
	Vp_vx_minus = shift_and_interp(2, -1)
	Vp_vy_plus = shift_and_interp(3, +1)
	Vp_vy_minus = shift_and_interp(3, -1)
	grad_vx = (Vp_vx_plus - Vp_vx_minus) / dvx
	grad_vy = (Vp_vy_plus - Vp_vy_minus) / dvy

	return V_interp, grad_vx, grad_vy


def _interp_scalar(V: np.ndarray, grid: Grid4D, coords):
	rx, ry, rvx, rvy = grid.axes
	idxs = []
	ws = []
	for axis_vals, val in zip((rx, ry, rvx, rvy), coords):
		i = int(np.clip(np.searchsorted(axis_vals, val) - 1, 0, len(axis_vals) - 2))
		idxs.append(i)
		w = (val - axis_vals[i]) / (axis_vals[i+1] - axis_vals[i])
		ws.append(w)
	V_interp = 0.0
	wx, wy, wvx, wvy = ws
	for b in range(16):
		ix = idxs[0] + ((b >> 0) & 1)
		iy = idxs[1] + ((b >> 1) & 1)
		ivx = idxs[2] + ((b >> 2) & 1)
		ivy = idxs[3] + ((b >> 3) & 1)
		cx = wx if (b & 1) else (1 - wx)
		cy = wy if (b & 2) else (1 - wy)
		cvx = wvx if (b & 4) else (1 - wvx)
		cvy = wvy if (b & 8) else (1 - wvy)
		V_interp += V[ix, iy, ivx, ivy] * cx * cy * cvx * cvy
	return V_interp


def simulate_closed_loop(V: np.ndarray, grid: Grid4D, state0: np.ndarray, a_p_max: float, a_e_max: float, dt: float, steps: int, capture_radius: float) -> dict:
	state = state0.astype(float).copy()
	traj = [state.copy()]
	for k in range(steps):
		_, gvx, gvy = _interp_central_gradients(V, grid, state)
		# Feedback controls
		ap, ae = optimal_controls_from_gradient(gvx, gvy, a_p_max, a_e_max)
		ap = ap.reshape(2)
		ae = ae.reshape(2)
		# Relative dynamics f(x) = [v_rel, a_e - a_p]
		def f(x):
			return np.array([x[2], x[3], ae[0] - ap[0], ae[1] - ap[1]], dtype=float)
		# RK4
		k1 = f(state)
		k2 = f(state + 0.5 * dt * k1)
		k3 = f(state + 0.5 * dt * k2)
		k4 = f(state + dt * k3)
		state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
		traj.append(state.copy())
		# stop if captured
		if np.sqrt(state[0]*state[0] + state[1]*state[1]) <= capture_radius:
			break
	return {"traj": np.array(traj), "steps": len(traj)-1}