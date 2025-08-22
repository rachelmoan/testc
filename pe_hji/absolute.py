import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from .grid import Grid4D
from .hji import solve_hji_lax_friedrichs
from .controls import optimal_controls_from_gradient


@dataclass
class AbsoluteGame:
	"""
	Absolute-state API for 2D pursuer-evader with double-integrator agents.

	This wrapper exposes only absolute states externally, but internally uses the
	relative-state value function V(r_x,r_y,r_vx,r_vy) to compute feedback.

	Absolute state vectors use ordering [x, y, vx, vy] for each agent.
	"""
	grid_rel: Grid4D
	capture_radius: float
	a_p_max: float
	a_e_max: float
	V: Optional[np.ndarray] = None

	def solve(self, t_max: float, dt: float, V0: Optional[np.ndarray] = None, verbose: bool = True) -> np.ndarray:
		"""
		Solve the relative-state HJI and store the value function for feedback.
		Returns V for convenience.
		"""
		self.V = solve_hji_lax_friedrichs(
			self.grid_rel,
			capture_radius=self.capture_radius,
			a_p_max=self.a_p_max,
			a_e_max=self.a_e_max,
			t_max=t_max,
			dt=dt,
			V0=V0,
			verbose=verbose,
		)
		return self.V

	def _relative_state(self, p_abs: np.ndarray, e_abs: np.ndarray) -> np.ndarray:
		return np.array([
			e_abs[0] - p_abs[0],
			e_abs[1] - p_abs[1],
			e_abs[2] - p_abs[2],
			e_abs[3] - p_abs[3],
		], dtype=float)

	def _interp_grad_v(self, state_rel: np.ndarray) -> np.ndarray:
		"""
		Interpolate the gradient wrt velocity components at a relative state.
		Returns gradient vector [dV/d(r_vx), dV/d(r_vy)].
		"""
		# Local central differences in index-space around the query
		dvx = self.grid_rel.dr[2]
		dvy = self.grid_rel.dr[3]
		def interp(coords):
			from .sim import _interp_scalar  # reuse scalar interpolation helper
			return float(_interp_scalar(self.V, self.grid_rel, coords))
		coords = state_rel.tolist()
		c = coords.copy()
		c[2] = np.clip(coords[2] + 0.5 * dvx, self.grid_rel.axes[2][0], self.grid_rel.axes[2][-1]); Vpx = interp(c)
		c[2] = np.clip(coords[2] - 0.5 * dvx, self.grid_rel.axes[2][0], self.grid_rel.axes[2][-1]); Vmx = interp(c)
		gx = (Vpx - Vmx) / dvx
		c = coords.copy()
		c[3] = np.clip(coords[3] + 0.5 * dvy, self.grid_rel.axes[3][0], self.grid_rel.axes[3][-1]); Vpy = interp(c)
		c[3] = np.clip(coords[3] - 0.5 * dvy, self.grid_rel.axes[3][0], self.grid_rel.axes[3][-1]); Vmy = interp(c)
		gy = (Vpy - Vmy) / dvy
		return np.array([gx, gy], dtype=float)

	def feedback_controls(self, p_abs: np.ndarray, e_abs: np.ndarray) -> Dict[str, np.ndarray]:
		"""
		Compute feedback accelerations (a_p, a_e) from absolute states using V.
		"""
		assert self.V is not None, "Call solve(...) first to compute V"
		x_rel = self._relative_state(p_abs, e_abs)
		grad_v = self._interp_grad_v(x_rel)
		a_p, a_e = optimal_controls_from_gradient(grad_v[0], grad_v[1], self.a_p_max, self.a_e_max)
		return {"a_p": a_p.reshape(2), "a_e": a_e.reshape(2)}

	def step(self, p_abs: np.ndarray, e_abs: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
		"""
		Advance absolute states by one step using RK4 under feedback controls.
		Returns new states and the controls applied.
		"""
		ctrl = self.feedback_controls(p_abs, e_abs)
		ap = ctrl["a_p"]; ae = ctrl["a_e"]

		def f_p(x):
			return np.array([x[2], x[3], ap[0], ap[1]], dtype=float)
		def f_e(x):
			return np.array([x[2], x[3], ae[0], ae[1]], dtype=float)

		k1p = f_p(p_abs); k1e = f_e(e_abs)
		k2p = f_p(p_abs + 0.5 * dt * k1p); k2e = f_e(e_abs + 0.5 * dt * k1e)
		k3p = f_p(p_abs + 0.5 * dt * k2p); k3e = f_e(e_abs + 0.5 * dt * k2e)
		k4p = f_p(p_abs + dt * k3p);      k4e = f_e(e_abs + dt * k3e)

		p_next = p_abs + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
		e_next = e_abs + (dt/6.0) * (k1e + 2*k2e + 2*k3e + k4e)
		return {"p": p_next, "e": e_next, "a_p": ap, "a_e": ae}

	def simulate(self, p0: np.ndarray, e0: np.ndarray, dt: float, steps: int, t_max: Optional[float] = None, v_p_bounds=None, v_e_bounds=None) -> Dict[str, np.ndarray]:
		"""
		Simulate absolute trajectories using HJI feedback defined by V.
		Stops on capture (||r|| ≤ R_c) or time horizon.
		"""
		assert self.V is not None, "Call solve(...) first to compute V"
		p = np.array(p0, dtype=float).copy(); e = np.array(e0, dtype=float).copy()
		traj_p = [p.copy()] ; traj_e = [e.copy()]
		T = 0.0; outcome = None
		for k in range(steps):
			res = self.step(p, e, dt)
			p, e = res["p"], res["e"]
			# clip velocities if requested
			if v_p_bounds is not None:
				p[2] = float(np.clip(p[2], v_p_bounds[0], v_p_bounds[1]))
				p[3] = float(np.clip(p[3], v_p_bounds[0], v_p_bounds[1]))
			if v_e_bounds is not None:
				e[2] = float(np.clip(e[2], v_e_bounds[0], v_e_bounds[1]))
				e[3] = float(np.clip(e[3], v_e_bounds[0], v_e_bounds[1]))
			traj_p.append(p.copy()); traj_e.append(e.copy())
			T += dt
			# capture check
			rx, ry = e[0]-p[0], e[1]-p[1]
			if np.hypot(rx, ry) <= self.capture_radius:
				outcome = 'pursuer_captures'; break
			if t_max is not None and T >= t_max:
				outcome = 'evader_escapes'; break
		if outcome is None:
			outcome = 'evader_escapes'
		traj_p = np.array(traj_p); traj_e = np.array(traj_e)
		traj_rel = np.column_stack((traj_e[:,0]-traj_p[:,0], traj_e[:,1]-traj_p[:,1], traj_e[:,2]-traj_p[:,2], traj_e[:,3]-traj_p[:,3]))
		return {"traj_p": traj_p, "traj_e": traj_e, "traj_rel": traj_rel, "T": T, "steps": len(traj_p)-1, "outcome": outcome}