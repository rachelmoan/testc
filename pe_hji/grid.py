import numpy as np
from dataclasses import dataclass


@dataclass
class Grid4D:
	"""
	Structured 4D grid for relative-state HJI on a 2D double integrator.

	State ordering and axes:
	- Position difference r = [r_x, r_y] = e_pos - p_pos
	- Relative velocity v_rel = [r_vx, r_vy] = e_vel - p_vel
	- The grid stores axes for (r_x, r_y, r_vx, r_vy) in that order.

	Attributes
	- r_bounds: tuple[float, float], symmetric or asymmetric bounds for each position axis (same for x and y)
	- v_bounds: tuple[float, float], bounds for each velocity axis (same for vx and vy)
	- n_r: number of samples along r_x and r_y
	- n_v: number of samples along r_vx and r_vy

	Derived
	- axes: 4-tuple of numpy arrays for each axis
	- dr: grid spacings (Δr_x, Δr_y, Δr_vx, Δr_vy)
	- shape: (n_r, n_r, n_v, n_v)
	"""
	# rx, ry, rvx, rvy axes
	r_bounds: tuple
	v_bounds: tuple
	n_r: int
	n_v: int

	def __post_init__(self):
		rx = np.linspace(self.r_bounds[0], self.r_bounds[1], self.n_r)
		ry = np.linspace(self.r_bounds[0], self.r_bounds[1], self.n_r)
		rvx = np.linspace(self.v_bounds[0], self.v_bounds[1], self.n_v)
		rvy = np.linspace(self.v_bounds[0], self.v_bounds[1], self.n_v)
		self.axes = (rx, ry, rvx, rvy)
		self.dr = (rx[1] - rx[0], ry[1] - ry[0], rvx[1] - rvx[0], rvy[1] - rvy[0])
		self.shape = (self.n_r, self.n_r, self.n_v, self.n_v)

	def mesh(self):
		"""
		Return 4D meshgrids (with 'ij' indexing) for (r_x, r_y, r_vx, r_vy).
		Useful for broadcasting Hamiltonian evaluations.
		"""
		return np.meshgrid(*self.axes, indexing='ij')

	def signed_distance_capture(self, capture_radius: float):
		"""
		Signed distance to the capture set in position subspace.

		Target/capture set T = { (r, v): ||r|| <= R_c }. We define
		φ(r, v) = ||r|| - R_c, independent of velocity, so φ < 0 inside capture.
		This is used as the initial terminal condition V(x, 0) = φ(x) for backward
		reachability (time-to-go) HJI when evolving forward in our discrete steps.
		"""
		rx, ry, rvx, rvy = self.mesh()
		r_norm = np.sqrt(rx**2 + ry**2)
		return r_norm - capture_radius

	def finite_differences(self, V: np.ndarray):
		"""
		Compute forward and backward first-order finite differences along each axis.

		We return lists [D+_rx, D+_ry, D+_rvx, D+_rvy] and [D-_rx, ...]. One-sided
		boundary stencils are used at the domain edges to maintain array shapes.
		These are used to form central gradients and Lax–Friedrichs dissipation.
		"""
		# Returns forward and backward diffs for each axis
		Dplus = []
		Dminus = []
		for axis, dx in enumerate(self.dr):
			Dplus_axis = (np.roll(V, -1, axis=axis) - V) / dx
			Dplus_axis = np.take(Dplus_axis, [-1], axis=axis, mode='clip') * 0 + Dplus_axis  # keep shape
			Dplus_axis = Dplus_axis.copy()
			# enforce one-sided at upper boundary by zeroing the last forward diff
			idx = [slice(None)] * V.ndim
			idx[axis] = -1
			Dplus_axis[tuple(idx)] = (V[tuple(idx)] - np.take(V, [-2], axis=axis).squeeze(axis)) / dx

			Dminus_axis = (V - np.roll(V, 1, axis=axis)) / dx
			# enforce one-sided at lower boundary
			idx[axis] = 0
			Dminus_axis[tuple(idx)] = (np.take(V, [1], axis=axis).squeeze(axis) - V[tuple(idx)]) / dx

			Dplus.append(Dplus_axis)
			Dminus.append(Dminus_axis)
		return Dplus, Dminus