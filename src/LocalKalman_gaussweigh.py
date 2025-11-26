"""
Local 3D Kalman filter with Gaussian-weighted observation operator. This
variant spreads a single measurement across a local window using a normalized
Gaussian kernel centered at the observation.
"""
import numpy as np
from matplotlib import pyplot as plt

class LocalKalmanFilter3D:
    def __init__(self, initial_concentration, global_variance=10.0, process_noise=0.1, 
                 measurement_noise=1.0, local_window=(8, 8, 3), 
                 decay_radius=4.0, weight_threshold=1e-3):
        self.concentration = initial_concentration.copy()
        self.shape = self.concentration.shape
        self.nx, self.ny, self.nz = self.shape
        self.global_variance = global_variance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.window = local_window
        self.decay_radius = decay_radius
        self.weight_threshold = weight_threshold

    def _gaussian_kernel(self, shape, center=None):
        """Generate a 3D Gaussian kernel normalized to sum to 1."""
        if center is None:
            center = [(s - 1) / 2 for s in shape]
        xx, yy, zz = np.meshgrid(
            np.arange(shape[0]) - center[0],
            np.arange(shape[1]) - center[1],
            np.arange(shape[2]) - center[2],
            indexing='ij'
        )
        distance_sq = xx**2 + yy**2 + zz**2
        sigma = self.decay_radius / 3
        kernel = np.exp(-distance_sq / (2 * sigma**2))
        kernel[kernel < self.weight_threshold] = 0
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[tuple(center)] = 1.0
        return kernel

    def _get_local_region(self, x, y, z):
        """Return local window slices and the kernel center offset."""
        wx, wy, wz = self.window
        x_min = max(0, x - wx // 2)
        x_max = min(self.nx, x + wx // 2 + 1)
        y_min = max(0, y - wy // 2)
        y_max = min(self.ny, y + wy // 2 + 1)
        z_min = max(0, z - wz // 2)
        z_max = min(self.nz, z + wz // 2 + 1)
        center_offset = (x - x_min, y - y_min, z - z_min)
        return (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)), center_offset

    def update(self, measurement_points, measurement_values):
        """Update using single or multiple points with Gaussian-weighted H."""
        
        points = np.atleast_2d(measurement_points)
        values = np.atleast_1d(measurement_values)
        if len(points) != len(values):
            raise ValueError("measurement_points和measurement_values长度不匹配")

        for (x, y, z), z_meas in zip(points, values):
            (local_slice, center_offset) = self._get_local_region(x, y, z)
            local_state = self.concentration[local_slice].flatten()
            n = local_state.size

            kernel_shape = (
                local_slice[0].stop - local_slice[0].start,
                local_slice[1].stop - local_slice[1].start,
                local_slice[2].stop - local_slice[2].start
            )
            H = self._gaussian_kernel(kernel_shape, center=center_offset).flatten().reshape(1, -1)

            P = np.eye(n) * self.global_variance + np.eye(n) * self.process_noise
            S = H @ P @ H.T + self.measurement_noise
            K = P @ H.T / S
            innovation = z_meas - H @ local_state
            local_state += K.flatten() * innovation
            self.concentration[local_slice] = local_state.reshape(self.concentration[local_slice].shape)

    def get_concentration(self):
        """Return the current concentration field."""
        return self.concentration

if __name__ == "__main__":
    x_dim, y_dim, z_dim = 140, 90, 10
    initial_concentration = np.ones((x_dim, y_dim, z_dim))
    
    kf = LocalKalmanFilter3D(
        initial_concentration, 
        global_variance=1.0,
        process_noise=0.1,
        measurement_noise=1.0,
        local_window=(8, 8, 3),
        decay_radius=2.0,
        weight_threshold=1e-3
    )
    
    
    
    measurement_points = np.array([
        [50, 50, 0],
        [120, 60, 0]
    ])
    measurement_values = np.array([0, 0])
    for i in range(3):
        kf.update([20, 30, 0], 0.0)
        kf.update(measurement_points, measurement_values)
    
    final_concentration = kf.get_concentration()
    plt.imshow(final_concentration[:, :, 0], cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Final Concentration at z=0")
    plt.show()
