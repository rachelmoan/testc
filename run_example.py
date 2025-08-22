import numpy as np
from pe_hji import Grid4D, solve_hji_lax_friedrichs, simulate_closed_loop, plot_value_slice, plot_trajectory, plot_pursuer_evader, save_step_frames


def main():
	# Agent velocity bounds
	v_p_bounds = (-1.5, 1.5)
	v_e_bounds = (-0.8, 0.8)
	# Relative velocity bounds for grid should cover possible differences
	rv_bounds = (v_e_bounds[0]-v_p_bounds[1], v_e_bounds[1]-v_p_bounds[0])

	# Problem setup
	r_bounds = (-5.0, 5.0)
	n_r = 21
	n_v = 21
	grid = Grid4D(r_bounds=r_bounds, v_bounds=rv_bounds, n_r=n_r, n_v=n_v)

	capture_radius = 0.5
	a_p_max = 1.0
	a_e_max = 0.8
	print(f"Using v_p_bounds={v_p_bounds}, v_e_bounds={v_e_bounds}, relative v bounds={rv_bounds}")

	# Solve HJI for a small horizon for speed
	V = solve_hji_lax_friedrichs(grid, capture_radius=capture_radius, a_p_max=a_p_max, a_e_max=a_e_max, t_max=0.8, dt=0.05, verbose=True)

	# Simulate closed-loop from a sample initial relative state [rx, ry, rvx, rvy]
	state0 = np.array([3.0, 2.0, -0.5, 0.2])
	sim = simulate_closed_loop(V, grid, state0, a_p_max=a_p_max, a_e_max=a_e_max, dt=0.05, steps=2000, capture_radius=capture_radius, t_max=10.0, v_p_bounds=v_p_bounds, v_e_bounds=v_e_bounds)

	traj = sim["traj"]
	print(f"Outcome: {sim['outcome']} at T={sim['T']:.2f}s, steps={sim['steps']}")
	print(f"Final r = ({traj[-1,0]:.2f}, {traj[-1,1]:.2f}), |r| = {np.linalg.norm(traj[-1,:2]):.3f}")
	print("First 3 states:")
	print(traj[:3])

	# Plots
	plot_value_slice(V, grid, rvx0=0.0, rvy0=0.0, capture_radius=capture_radius, filename='/workspace/figs/value_slice_rvx0_rvy0.png')
	plot_trajectory(traj, capture_radius=capture_radius, filename='/workspace/figs/trajectory_xy.png')
	plot_pursuer_evader(sim["traj_p"], sim["traj_e"], capture_radius=capture_radius, filename='/workspace/figs/pursuer_evader_xy.png')
	# Per-step frames
	save_step_frames(sim["traj_p"], sim["traj_e"], capture_radius=capture_radius, out_dir='/workspace/figs/frames')
	print('Saved figures to /workspace/figs/ and frames to /workspace/figs/frames')


if __name__ == "__main__":
	main()