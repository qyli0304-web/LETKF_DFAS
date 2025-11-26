"""
Local windowed 3D Kalman filter for updating a concentration field near
measurement points. Covariance is handled locally per window to reduce cost.
"""
import numpy as np
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import inv

class LocalKalmanFilter3D:
    def __init__(self, initial_concentration, global_variance=10.0, process_noise=0.1, measurement_noise=1.0, local_window=(5,5,3)):
        self.concentration = initial_concentration.copy()
        self.shape = self.concentration.shape
        self.nx, self.ny, self.nz = self.shape
        self.global_variance = global_variance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.window = local_window

    def _get_local_region(self, x, y, z):
        """Return slice bounds for the local window around (x, y, z)."""
        wx, wy, wz = self.window
        x_min = max(0, x - wx//2)
        x_max = min(self.nx, x + wx//2 + 1)
        y_min = max(0, y - wy//2)
        y_max = min(self.ny, y + wy//2 + 1)
        z_min = max(0, z - wz//2)
        z_max = min(self.nz, z + wz//2 + 1)
        return (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))

    def predict(self):
        """Prediction step (identity dynamics assumed)."""
        pass

    def update(self, measurement_points, measurement_values):
        points = np.atleast_2d(measurement_points)
        values = np.atleast_1d(measurement_values)
        if len(points) != len(values):
            raise ValueError("measurement_points and measurement_values length do not match")
        for (x, y, z), z_meas in zip(points, values):
            local_slice = self._get_local_region(x, y, z)
            local_state = self.concentration[local_slice].flatten()

            n = local_state.size
            H = np.zeros((1, n))

            
            lx, ly, lz = x - local_slice[0].start, y - local_slice[1].start, z - local_slice[2].start
            local_index = lx + ly * (local_slice[0].stop - local_slice[0].start) + lz * (local_slice[0].stop - local_slice[0].start) * (local_slice[1].stop - local_slice[1].start)
            H[0, local_index] = 1

            P = np.eye(n) * self.global_variance
            Q = np.eye(n) * self.process_noise
            R = self.measurement_noise

            P = P + Q

            S = H @ P @ H.T + R
            K = P @ H.T / S 

            innovation = z_meas - H @ local_state
            local_state = local_state + (K.flatten() * innovation)
            P = (np.eye(n) - K @ H) @ P

            self.concentration[local_slice] = local_state.reshape(self.concentration[local_slice].shape)

    def get_concentration(self):
        """Return the current concentration field."""
        return self.concentration


if __name__ == "__main__":
    x_dim, y_dim, z_dim = 140, 90, 10
    initial_concentration = np.zeros((x_dim, y_dim, z_dim))
    
    
    kf = LocalKalmanFilter3D(
        initial_concentration, 
        global_variance=10.0,
        process_noise=0.1,
        measurement_noise=1.0,
        local_window=(30, 30, 3)
    )
    
    measurement_points = np.array([
        [20, 30, 0],
        [50, 50, 0],
        [120, 60, 0]
    ])
    measurement_values = np.array([5.0, 10.0, 15.0])
    
    for _ in range(15):
        kf.update(measurement_points, measurement_values)

    final_concentration = kf.get_concentration()
    print("Final concentration shape:", final_concentration.shape)
    show = final_concentration[:, :, 0]
    from matplotlib import pyplot as plt
    plt.imshow(show, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Final Concentration at z=0")
    plt.xlabel("X Dimension")
    plt.ylabel("Y Dimension")
    plt.show()
    print("Final shape:", final_concentration.shape)
