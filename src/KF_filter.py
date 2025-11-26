"""
Basic 3D Kalman Filter for concentration fields represented on a grid.
Provides predict and update steps with point measurements on the grid.
"""
import numpy as np

class KalmanFilter3D:
    def __init__(self, initial_concentration, initial_covariance, process_noise, measurement_noise):
        """Initialize the 3D Kalman filter state and covariances.

        Parameters
        ----------
        initial_concentration : np.ndarray (x, y, z)
        initial_covariance : np.ndarray ((x*y*z), (x*y*z))
        process_noise : np.ndarray ((x*y*z), (x*y*z))
        measurement_noise : float or np.ndarray
        """
        self.concentration = initial_concentration.flatten()  
        self.covariance = initial_covariance  
        
        x_dim, y_dim, z_dim = initial_concentration.shape
        n_states = x_dim * y_dim * z_dim
        self.F = np.eye(n_states)  
        
        self.Q = process_noise  
        
        if np.isscalar(measurement_noise):
            self.R = measurement_noise
        else:
            self.R = measurement_noise
        
        self.dimensions = initial_concentration.shape
    
    def predict(self):
        """Prediction step."""
        self.concentration = self.F @ self.concentration
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.concentration.reshape(self.dimensions)
    
    def update(self, measurement_points, measurement_values):
        """Update step with point measurements.

        Parameters
        ----------
        measurement_points : np.ndarray (m, 3)
            Grid indices of measurements (x, y, z).
        measurement_values : np.ndarray (m,)
        """
        m = measurement_points.shape[0] 
        
        H = np.zeros((m, len(self.concentration)))
        x_dim, y_dim, z_dim = self.dimensions
        
        for i, (x, y, z) in enumerate(measurement_points):
            idx = x + y * x_dim + z * x_dim * y_dim
            H[i, idx] = 1
        
        if np.isscalar(self.R):
            S = H @ self.covariance @ H.T + self.R * np.eye(m)
        else:
            S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        innovation = measurement_values - H @ self.concentration
        self.concentration = self.concentration + K @ innovation
        self.covariance = (np.eye(len(self.concentration)) - K @ H) @ self.covariance
        
        return self.concentration.reshape(self.dimensions)
    
    def get_concentration(self):
        """Return the current concentration estimate reshaped to 3D."""
        return self.concentration.reshape(self.dimensions)


if __name__ == "__main__":
    x_dim, y_dim, z_dim = 10, 10, 10
    
    initial_concentration = np.zeros((x_dim, y_dim, z_dim))
    
    initial_covariance = 10 * np.eye(x_dim * y_dim * z_dim)
    
    process_noise = 0.1 * np.eye(x_dim * y_dim * z_dim)
    
    measurement_noise = 1.0  
    
    kf = KalmanFilter3D(initial_concentration, initial_covariance, process_noise, measurement_noise)
    
    measurement_points = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    measurement_values = np.array([10.2, 15.3, 20.1])
    
    for _ in range(5):  
        pred = kf.predict()
        updated = kf.update(measurement_points, measurement_values)
    
    final_concentration = kf.get_concentration()
    print("Final concentration field shape:", final_concentration.shape)