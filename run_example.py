import numpy as np
from pe_hji import Grid4D, Grid6D, solve_hji_lax_friedrichs, solve_hji_lax_friedrichs_6d, simulate_closed_loop, simulate_closed_loop_6d, plot_value_slice, plot_value_slice_6d, plot_trajectory, plot_pursuer_evader, plot_pursuer_evader_3d


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
	plot_pursuer_evader(sim["traj_p"], sim["traj_e"], capture_radius=capture_radius, filename='/workspace/figs/pursuer_evader_xy.png')
	print('Saved figures to /workspace/figs/')

	# 3D example (coarse grid due to 6D state size)
	r_bounds3 = (-3.0, 3.0)
	v_bounds3 = (-1.5, 1.5)
	n_r3 = 9
	n_v3 = 7
	grid6 = Grid6D(r_bounds=r_bounds3, v_bounds=v_bounds3, n_r=n_r3, n_v=n_v3)

	capture_radius3 = 0.7
	a_p_max3 = 1.0
	a_e_max3 = 0.8

	V6 = solve_hji_lax_friedrichs_6d(grid6, capture_radius=capture_radius3, a_p_max=a_p_max3, a_e_max=a_e_max3, t_max=0.4, dt=0.05, verbose=True)

	state0_6d = np.array([2.0, -1.0, 1.2, -0.2, 0.1, 0.15])
	sim3 = simulate_closed_loop_6d(V6, grid6, state0_6d, a_p_max=a_p_max3, a_e_max=a_e_max3, dt=0.05, steps=120, capture_radius=capture_radius3)
	traj3 = sim3["traj"]
	print(f"3D Simulated {sim3['steps']} steps. Final |r| = {np.linalg.norm(traj3[-1,:3]):.3f}")
	plot_value_slice_6d(V6, grid6, rvx0=0.0, rvy0=0.0, rvz0=0.0, rz0=0.0, capture_radius=capture_radius3, filename='/workspace/figs/value_slice6d_rv0_rz0.png')
	plot_pursuer_evader_3d(sim3["traj_p"], sim3["traj_e"], capture_radius=capture_radius3, filename='/workspace/figs/pursuer_evader_xyz.png')


if __name__ == "__main__":
	main()