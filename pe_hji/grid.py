import numpy as np
from dataclasses import dataclass


@dataclass
class Grid4D:
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
		return np.meshgrid(*self.axes, indexing='ij')

	def signed_distance_capture(self, capture_radius: float):
		rx, ry, rvx, rvy = self.mesh()
		r_norm = np.sqrt(rx**2 + ry**2)
		return r_norm - capture_radius

	def finite_differences(self, V: np.ndarray):
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


@dataclass
class Grid6D:
	# rx, ry, rz, rvx, rvy, rvz axes
	r_bounds: tuple
	v_bounds: tuple
	n_r: int
	n_v: int

	def __post_init__(self):
		rx = np.linspace(self.r_bounds[0], self.r_bounds[1], self.n_r)
		ry = np.linspace(self.r_bounds[0], self.r_bounds[1], self.n_r)
		rz = np.linspace(self.r_bounds[0], self.r_bounds[1], self.n_r)
		rvx = np.linspace(self.v_bounds[0], self.v_bounds[1], self.n_v)
		rvy = np.linspace(self.v_bounds[0], self.v_bounds[1], self.n_v)
		rvz = np.linspace(self.v_bounds[0], self.v_bounds[1], self.n_v)
		self.axes = (rx, ry, rz, rvx, rvy, rvz)
		self.dr = (
			rx[1] - rx[0],
			ry[1] - ry[0],
			rz[1] - rz[0],
			rvx[1] - rvx[0],
			rvy[1] - rvy[0],
			rvz[1] - rvz[0],
		)
		self.shape = (self.n_r, self.n_r, self.n_r, self.n_v, self.n_v, self.n_v)

	def mesh(self):
		return np.meshgrid(*self.axes, indexing='ij')

	def signed_distance_capture(self, capture_radius: float):
		rx, ry, rz, rvx, rvy, rvz = self.mesh()
		r_norm = np.sqrt(rx**2 + ry**2 + rz**2)
		return r_norm - capture_radius

	def finite_differences(self, V: np.ndarray):
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