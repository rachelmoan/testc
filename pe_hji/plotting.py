import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid import Grid4D


def _ensure_dir(path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_value_slice(V: np.ndarray, grid: Grid4D, rvx0: float, rvy0: float, capture_radius: float, filename: str):
	rx, ry, rvx, rvy = grid.axes
	ivx = int(np.clip(np.searchsorted(rvx, rvx0), 0, len(rvx)-1))
	ivy = int(np.clip(np.searchsorted(rvy, rvy0), 0, len(rvy)-1))
	Vs = V[:, :, ivx, ivy]
	_extent = [rx[0], rx[-1], ry[0], ry[-1]]
	_ensure_dir(filename)
	plt.figure(figsize=(5,4), dpi=120)
	im = plt.imshow(Vs.T, origin='lower', extent=_extent, aspect='equal', cmap='viridis')
	plt.colorbar(im, fraction=0.046, pad=0.04, label='V')
	circle = plt.Circle((0,0), capture_radius, color='r', fill=False, linestyle='--', linewidth=1.5)
	plt.gca().add_patch(circle)
	plt.title(f"Value slice at rvx={rvx[ivx]:.2f}, rvy={rvy[ivy]:.2f}")
	plt.xlabel('r_x')
	plt.ylabel('r_y')
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def plot_trajectory(traj: np.ndarray, capture_radius: float, filename: str):
	_ensure_dir(filename)
	plt.figure(figsize=(5,4), dpi=120)
	rx = traj[:,0]
	ry = traj[:,1]
	plt.plot(rx, ry, '-k', linewidth=1.5, label='trajectory')
	plt.plot(rx[0], ry[0], 'go', label='start')
	plt.plot(rx[-1], ry[-1], 'ro', label='end')
	circle = plt.Circle((0,0), capture_radius, color='r', fill=False, linestyle='--', linewidth=1.5, label='capture')
	plt.gca().add_patch(circle)
	plt.axis('equal')
	plt.xlabel('r_x')
	plt.ylabel('r_y')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()