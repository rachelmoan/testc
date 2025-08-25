import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grid import Grid4D
from .grid import Grid6D


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


def plot_pursuer_evader(traj_p: np.ndarray, traj_e: np.ndarray, capture_radius: float, filename: str):
	_ensure_dir(filename)
	plt.figure(figsize=(5,4), dpi=120)
	plt.plot(traj_p[:,0], traj_p[:,1], '-b', linewidth=2.0, label='pursuer')
	plt.plot(traj_e[:,0], traj_e[:,1], '-r', linewidth=2.0, label='evader')
	plt.plot(traj_p[0,0], traj_p[0,1], 'bo', label='p start')
	plt.plot(traj_e[0,0], traj_e[0,1], 'ro', label='e start')
	plt.plot(traj_p[-1,0], traj_p[-1,1], 'b^', label='p end')
	plt.plot(traj_e[-1,0], traj_e[-1,1], 'r^', label='e end')
	circle = plt.Circle((0,0), capture_radius, color='k', fill=False, linestyle='--', linewidth=1.0, label='capture')
	plt.gca().add_patch(circle)
	plt.axis('equal')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def plot_value_slice_6d(V: np.ndarray, grid: Grid6D, rvx0: float, rvy0: float, rvz0: float, rz0: float, capture_radius: float, filename: str):
	rx, ry, rz, rvx, rvy, rvz = grid.axes
	ivx = int(np.clip(np.searchsorted(rvx, rvx0), 0, len(rvx)-1))
	ivy = int(np.clip(np.searchsorted(rvy, rvy0), 0, len(rvy)-1))
	ivz = int(np.clip(np.searchsorted(rvz, rvz0), 0, len(rvz)-1))
	iz  = int(np.clip(np.searchsorted(rz, rz0), 0, len(rz)-1))
	Vs = V[:, :, iz, ivx, ivy, ivz]
	_extent = [rx[0], rx[-1], ry[0], ry[-1]]
	_ensure_dir(filename)
	plt.figure(figsize=(5,4), dpi=120)
	im = plt.imshow(Vs.T, origin='lower', extent=_extent, aspect='equal', cmap='viridis')
	plt.colorbar(im, fraction=0.046, pad=0.04, label='V')
	circle = plt.Circle((0,0), capture_radius, color='r', fill=False, linestyle='--', linewidth=1.5)
	plt.gca().add_patch(circle)
	plt.title(f"V slice rv=({rvx[ivx]:.2f},{rvy[ivy]:.2f},{rvz[ivz]:.2f}), rz={rz[iz]:.2f}")
	plt.xlabel('r_x')
	plt.ylabel('r_y')
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def plot_pursuer_evader_3d(traj_p: np.ndarray, traj_e: np.ndarray, capture_radius: float, filename: str):
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	_ensure_dir(filename)
	fig = plt.figure(figsize=(6,5), dpi=120)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(traj_p[:,0], traj_p[:,1], traj_p[:,2], '-b', linewidth=2.0, label='pursuer')
	ax.plot(traj_e[:,0], traj_e[:,1], traj_e[:,2], '-r', linewidth=2.0, label='evader')
	ax.scatter(traj_p[0,0], traj_p[0,1], traj_p[0,2], c='b', marker='o', label='p start')
	ax.scatter(traj_e[0,0], traj_e[0,1], traj_e[0,2], c='r', marker='o', label='e start')
	ax.scatter(traj_p[-1,0], traj_p[-1,1], traj_p[-1,2], c='b', marker='^', label='p end')
	ax.scatter(traj_e[-1,0], traj_e[-1,1], traj_e[-1,2], c='r', marker='^', label='e end')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.legend(loc='best')
	plt.tight_layout()
	plt.savefig(filename)
	plt.close()