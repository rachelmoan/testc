pe_hji - Pursuit-Evasion HJI (2D and 3D aircraft, 1v1)

Quick start

1) Install
   pip install -r requirements.txt

2) Run 2D and 3D demos
   python /workspace/run_example.py

This computes a short-horizon value function V on a 4D grid [rx, ry, rvx, rvy], simulates closed-loop pursuit-evasion, and saves plots into /workspace/figs/.

The same script also runs a coarse 6D problem [rx, ry, rz, rvx, rvy, rvz] using a Lax–Friedrichs HJI solver with a Stackelberg Hamiltonian and acceleration-bounded controls aligned with the gradient of V wrt relative velocity. It saves a 2D slice of V and a 3D trajectory plot into /workspace/figs/.

API

- Grid4D, Grid6D: Regular grids for 4D/6D relative-state spaces
- solve_hji_lax_friedrichs, solve_hji_lax_friedrichs_6d: Isotropic LF HJI solvers
- simulate_closed_loop, simulate_closed_loop_6d: Feedback simulations with RK4 integration
- plotting: value slices and 2D/3D trajectory plots

Notes

- The 6D example is intentionally small for speed. Increase n_r and n_v cautiously as memory scales as O(n_r^3 n_v^3).