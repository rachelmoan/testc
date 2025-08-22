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


def simulate_closed_loop(V: np.ndarray, grid: Grid4D, state0: np.ndarray, a_p_max: float, a_e_max: float, dt: float, steps: int, capture_radius: float, p0=None, e0=None, t_max=None, v_p_bounds=None, v_e_bounds=None) -> dict:
	# Initialize relative state and absolute pursuer/evader states
	state = state0.astype(float).copy()
	if p0 is None:
		p_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
	else:
		p_state = np.array(p0, dtype=float).copy()
	if e0 is None:
		e_state = np.array([state[0], state[1], state[2], state[3]], dtype=float)
	else:
		e_state = np.array(e0, dtype=float).copy()
	# Clip initial velocities if bounds provided
	if v_p_bounds is not None:
		p_state[2] = float(np.clip(p_state[2], v_p_bounds[0], v_p_bounds[1]))
		p_state[3] = float(np.clip(p_state[3], v_p_bounds[0], v_p_bounds[1]))
	if v_e_bounds is not None:
		e_state[2] = float(np.clip(e_state[2], v_e_bounds[0], v_e_bounds[1]))
		e_state[3] = float(np.clip(e_state[3], v_e_bounds[0], v_e_bounds[1]))

	traj_rel = [np.array([e_state[0]-p_state[0], e_state[1]-p_state[1], e_state[2]-p_state[2], e_state[3]-p_state[3]], dtype=float)]
	traj_p = [p_state.copy()]
	traj_e = [e_state.copy()]
	outcome = None
	T = 0.0
	for k in range(steps):
		# Build relative state used for feedback
		state = np.array([e_state[0]-p_state[0], e_state[1]-p_state[1], e_state[2]-p_state[2], e_state[3]-p_state[3]], dtype=float)
		_, gvx, gvy = _interp_central_gradients(V, grid, state)
		# Feedback controls
		ap, ae = optimal_controls_from_gradient(gvx, gvy, a_p_max, a_e_max)
		ap = ap.reshape(2)
		ae = ae.reshape(2)
		# Absolute dynamics for pursuer and evader
		def f_p(x):
			return np.array([x[2], x[3], ap[0], ap[1]], dtype=float)
		def f_e(x):
			return np.array([x[2], x[3], ae[0], ae[1]], dtype=float)
		# RK4 with frozen controls over step
		k1p = f_p(p_state)
		k1e = f_e(e_state)
		k2p = f_p(p_state + 0.5 * dt * k1p)
		k2e = f_e(e_state + 0.5 * dt * k1e)
		k3p = f_p(p_state + 0.5 * dt * k2p)
		k3e = f_e(e_state + 0.5 * dt * k2e)
		k4p = f_p(p_state + dt * k3p)
		k4e = f_e(e_state + dt * k3e)
		p_state = p_state + (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
		e_state = e_state + (dt / 6.0) * (k1e + 2*k2e + 2*k3e + k4e)
		# Enforce per-agent velocity bounds if provided
		if v_p_bounds is not None:
			p_state[2] = float(np.clip(p_state[2], v_p_bounds[0], v_p_bounds[1]))
			p_state[3] = float(np.clip(p_state[3], v_p_bounds[0], v_p_bounds[1]))
		if v_e_bounds is not None:
			e_state[2] = float(np.clip(e_state[2], v_e_bounds[0], v_e_bounds[1]))
			e_state[3] = float(np.clip(e_state[3], v_e_bounds[0], v_e_bounds[1]))

		T += dt
		# Build new relative state
		state = np.array([e_state[0]-p_state[0], e_state[1]-p_state[1], e_state[2]-p_state[2], e_state[3]-p_state[3]], dtype=float)

		traj_rel.append(state.copy())
		traj_p.append(p_state.copy())
		traj_e.append(e_state.copy())
		# stop if captured
		if np.sqrt(state[0]*state[0] + state[1]*state[1]) <= capture_radius:
			outcome = 'pursuer_captures'
			break
		# stop if time exceeded
		if t_max is not None and T >= t_max:
			outcome = 'evader_escapes'
			break
	# If no outcome set, it's max steps without t_max -> treat as horizon reached
	if outcome is None:
		outcome = 'evader_escapes'
	return {"traj": np.array(traj_rel), "traj_rel": np.array(traj_rel), "traj_p": np.array(traj_p), "traj_e": np.array(traj_e), "steps": len(traj_rel)-1, "T": T, "outcome": outcome}