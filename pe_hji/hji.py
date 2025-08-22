import numpy as np
from .grid import Grid4D
from .dynamics import stackelberg_hamiltonian, bounds_hamiltonian_partials


def _central_gradient(Dplus, Dminus):
	# Returns central derivative for each axis and dissipation term (D+ - D-)/2
	grads = []
	diss = []
	for Dp, Dm in zip(Dplus, Dminus):
		grads.append(0.5 * (Dp + Dm))
		diss.append(0.5 * (Dp - Dm))
	return grads, diss


def solve_hji_lax_friedrichs(grid: Grid4D, capture_radius: float, a_p_max: float, a_e_max: float, t_max: float, dt: float, V0: np.ndarray = None, verbose: bool = True):
	# Initialize value function with signed distance to target set over position components
	if V0 is None:
		V = grid.signed_distance_capture(capture_radius)
	else:
		V = V0.copy()

	# Precompute alpha bounds for LF dissipation
	alpha1, alpha2, alpha3, alpha4 = bounds_hamiltonian_partials(grid.axes[2], grid.axes[3], a_p_max, a_e_max)

	n_steps = int(np.ceil(t_max / dt))
	for k in range(n_steps):
		Dplus, Dminus = grid.finite_differences(V)
		(grx, gry, grvx, grvy), (dissx, dissy, dissvx, dissvy) = _central_gradient(Dplus, Dminus)

		# Hamiltonian at central gradients
		# rvx,rvy fields depend only on axes, so broadcast via meshgrid
		rx, ry, rvx, rvy = grid.mesh()
		H = stackelberg_hamiltonian(rvx, rvy, grx, gry, grvx, grvy, a_p_max, a_e_max)

		# LF dissipation term
		H -= alpha1 * dissx + alpha2 * dissy + alpha3 * dissvx + alpha4 * dissvy

		V = V - dt * H

		if verbose and (k % max(1, n_steps // 10) == 0 or k == n_steps - 1):
			minV = float(np.min(V))
			maxV = float(np.max(V))
			print(f"HJI step {k+1}/{n_steps}: V in [{minV:.3f}, {maxV:.3f}]")

	return V