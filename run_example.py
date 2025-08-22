import numpy as np
from pe_hji import Grid4D, solve_hji_lax_friedrichs, simulate_closed_loop, plot_value_slice, plot_trajectory


def main():
	# Problem setup
	r_bounds = (-5.0, 5.0)
	v_bounds = (-2.0, 2.0)
	n_r = 21
	n_v = 21
	grid = Grid4D(r_bounds=r_bounds, v_bounds=v_bounds, n_r=n_r, n_v=n_v)

	capture_radius = 0.5
	a_p_max = 1.0
	a_e_max = 0.8

	# Solve HJI for a small horizon for speed
	V = solve_hji_lax_friedrichs(grid, capture_radius=capture_radius, a_p_max=a_p_max, a_e_max=a_e_max, t_max=0.8, dt=0.05, verbose=True)

	# Simulate closed-loop from a sample initial relative state [rx, ry, rvx, rvy]
	state0 = np.array([3.0, 2.0, -0.5, 0.2])
	sim = simulate_closed_loop(V, grid, state0, a_p_max=a_p_max, a_e_max=a_e_max, dt=0.05, steps=200, capture_radius=capture_radius)

	traj = sim["traj"]
	print(f"Simulated {sim['steps']} steps. Final r = ({traj[-1,0]:.2f}, {traj[-1,1]:.2f}), |r| = {np.linalg.norm(traj[-1,:2]):.3f}")
	print("First 3 states:")
	print(traj[:3])

	# Plots
	plot_value_slice(V, grid, rvx0=0.0, rvy0=0.0, capture_radius=capture_radius, filename='/workspace/figs/value_slice_rvx0_rvy0.png')
	plot_trajectory(traj, capture_radius=capture_radius, filename='/workspace/figs/trajectory_xy.png')
	print('Saved figures to /workspace/figs/')


if __name__ == "__main__":
	main()