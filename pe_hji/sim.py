import numpy as np
from typing import Tuple
from .grid import Grid4D
from .controls import optimal_controls_from_gradient


def _interp_gradients(V: np.ndarray, grid: Grid4D, state: np.ndarray) -> Tuple[float, float, float, float, float]:
	rx, ry, rvx, rvy = grid.axes
	coords = [state[0], state[1], state[2], state[3]]
	def central_diff(axis_idx: int, h: float):
		coords_plus = coords.copy(); coords_minus = coords.copy()
		coords_plus[axis_idx] = float(np.clip(coords_plus[axis_idx] + 0.5 * h, grid.axes[axis_idx][0], grid.axes[axis_idx][-1]))
		coords_minus[axis_idx] = float(np.clip(coords_minus[axis_idx] - 0.5 * h, grid.axes[axis_idx][0], grid.axes[axis_idx][-1]))
		Vp = _interp_scalar(V, grid, coords_plus)
		Vm = _interp_scalar(V, grid, coords_minus)
		return (Vp - Vm) / h
	V_here = _interp_scalar(V, grid, coords)
	grx = central_diff(0, grid.dr[0])
	gry = central_diff(1, grid.dr[1])
	gvx = central_diff(2, grid.dr[2])
	gvy = central_diff(3, grid.dr[3])
	return V_here, grx, gry, gvx, gvy


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


def simulate_closed_loop(V: np.ndarray, grid: Grid4D, state0: np.ndarray = None, a_p_max: float = 1.0, a_e_max: float = 1.0, dt: float = 0.05, steps: int = 1000, capture_radius: float = 0.5, p0=None, e0=None, t_max=None, v_p_bounds=None, v_e_bounds=None) -> dict:
	# Initialize absolute pursuer/evader from inputs; derive relative
	if p0 is None:
		p_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
	else:
		p_state = np.array(p0, dtype=float).copy()
	if e0 is None:
		if state0 is None:
			raise ValueError("Provide either (p0,e0) or relative state0")
		# derive evader from relative state if only p0 and state0 provided
		e_state = np.array([p_state[0] + state0[0], p_state[1] + state0[1], p_state[2] + state0[2], p_state[3] + state0[3]], dtype=float)
	else:
		e_state = np.array(e0, dtype=float).copy()
	# If state0 not provided, compute from p/e
	if state0 is None:
		state = np.array([e_state[0]-p_state[0], e_state[1]-p_state[1], e_state[2]-p_state[2], e_state[3]-p_state[3]], dtype=float)
	else:
		state = np.array(state0, dtype=float).copy()
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
		_, grx, gry, gvx, gvy = _interp_gradients(V, grid, state)
		# Feedback controls from gradient wrt velocity components
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