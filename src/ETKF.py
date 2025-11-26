"""
Ensemble Transform Kalman Filter (ETKF) utilities for 3D concentration fields.
Includes ETKF analysis, perturbation update, diagnostic metrics, and helpers.
"""
import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse import lil_matrix, eye
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
def analyze_xy_concentration(conc_field):
    """Analyze XY-plane patterns of a 3D concentration field.

    Parameters
    ----------
    conc_field : np.ndarray (x, y, z)
        3D concentration field.

    Returns
    -------
    tuple
        Number of peaks and qualitative concentration characterization.
    """
    
    xy_mean = np.mean(conc_field, axis=2)
    
    xy_smoothed = gaussian_filter(xy_mean, sigma=1)
    
    flattened = xy_smoothed.flatten()
    peaks, _ = find_peaks(flattened, prominence=np.std(flattened)/2)
    num_peaks = len(peaks)
    
    from scipy.stats import kurtosis
    kurt = kurtosis(flattened)
    if kurt > 3:
        concentration = "集中"
    else:
        concentration = "分散"
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(xy_mean, cmap='viridis')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(xy_smoothed, cmap='viridis')
    plt.colorbar()
    
    plt.subplot(133)
    hist, bins = np.histogram(flattened, bins=50)
    plt.plot(bins[:-1], hist)
    
    plt.tight_layout()
    plt.show()
    return num_peaks, concentration

def ETKF(ensemble, observations, obs_locations, R,
         prior_inflation: float = 1.0,
         posterior_inflation: float = 1.0,
         spread_restoration: str = 'none',
         rtps_alpha: float = 0.0,
         rtpp_alpha: float = 0.0,
         neg_innovation_boost: bool = False,
         neg_innovation_factor: float = 1.0):
    """Perform ETKF analysis on an ensemble of 3D fields.

    Parameters
    ----------
    ensemble : np.ndarray (n, x, y, z)
    observations : array-like (m, 1)
    obs_locations : array-like (m, 3)
        Grid indices of observation points.
    R : float or ndarray
        Observation error variance/covariance.

    Returns
    -------
    tuple
        Analysis ensemble (n, x, y, z) and analysis mean (x, y, z).
    """
    n, x, y, z = ensemble.shape
    ensemble_flat = ensemble.reshape(n, -1)
    mean = np.mean(ensemble_flat, axis=0)
    observations = np.array(observations).reshape(-1, 1)
    
    perturbations = (ensemble_flat - mean) / np.sqrt(n - 1)
    if prior_inflation != 1.0:
        perturbations = prior_inflation * perturbations
    
    H = np.zeros((len(obs_locations), x * y * z))
    for i, (xi, yi, zi) in enumerate(obs_locations):
        idx = xi * y * z + yi * z + zi
        H[i, idx] = 1
    
    HX = H @ perturbations.T
    
    R_inv = np.linalg.inv(R * np.eye(len(obs_locations)))
    C = HX.T @ R_inv @ HX
    
    w, V = np.linalg.eigh(C)
    
    T = V @ np.diag(1.0 / np.sqrt(1.0 + w))
    
    analysis_perturbations = perturbations.T @ T * np.sqrt(n - 1)  
    analysis_perturbations = analysis_perturbations.T  
    
    kalman_gain = perturbations.T @ HX.T @ np.linalg.inv(HX @ HX.T + R * np.eye(len(obs_locations)))
    innovation = observations - H @ mean.reshape(-1, 1)
    if neg_innovation_boost:
        sign_mask = (innovation < 0).astype(float)
        innovation = innovation * (1.0 + sign_mask * (neg_innovation_factor - 1.0))
    mean_analysis = mean.reshape(-1, 1) + kalman_gain @ innovation
    
    if posterior_inflation != 1.0:
        analysis_perturbations = posterior_inflation * analysis_perturbations

    prior_anomalies = (perturbations * np.sqrt(n - 1)).T
    prior_anomalies = prior_anomalies.T

    s = analysis_perturbations.shape[1]
    if spread_restoration.lower() == 'rtps':
        eps = 1e-8
        sigma_b = np.std(prior_anomalies, axis=0, ddof=1)
        sigma_a = np.std(analysis_perturbations, axis=0, ddof=1)
        factor = (1.0 - rtps_alpha) + rtps_alpha * (sigma_b / (sigma_a + eps))
        analysis_perturbations = analysis_perturbations * factor.reshape(1, s)
    elif spread_restoration.lower() == 'rtpp':
        analysis_perturbations = (1.0 - rtpp_alpha) * analysis_perturbations + rtpp_alpha * prior_anomalies

    analysis_ensemble_flat = mean_analysis.reshape(-1) + analysis_perturbations
    return analysis_ensemble_flat.reshape(n, x, y, z), mean_analysis.reshape(x, y, z)

def update_ETKF_PERTURBATION(ensemble, obs_locations, R,
                             prior_inflation: float = 1.0):
    """Update ETKF perturbations given observation locations and R."""
    n, x, y, z = ensemble.shape
    ensemble_flat = ensemble.reshape(n, -1)
    
    mean = np.mean(ensemble_flat, axis=0)
    perturbations = (ensemble_flat - mean) / np.sqrt(n - 1)
    if prior_inflation != 1.0:
        perturbations = prior_inflation * perturbations
    
    H = np.zeros((len(obs_locations), x * y * z))
    for i, (xi, yi, zi) in enumerate(obs_locations):
        idx = xi * y * z + yi * z + zi
        H[i, idx] = 1
    
    HX = H @ perturbations.T
    
    R_inv = np.linalg.inv(R * np.eye(len(obs_locations)))
    C = HX.T @ R_inv @ HX
    
    w, V = np.linalg.eigh(C)
    
    T = V @ np.diag(1.0 / np.sqrt(1.0 + w))
    analysis_perturbations = perturbations.T @ T * np.sqrt(n - 1)  
    analysis_perturbations = analysis_perturbations.T
    
    return analysis_perturbations.reshape(n, x, y, z)

def calculate_cv_field(array):
    """Compute coefficient-of-variation field across ensembles (safe at zero mean)."""
    mean_field = np.mean(array, axis=0)
    std_field = np.std(array, axis=0, ddof=1)  
    
    cv_field = np.zeros_like(mean_field)
    
    non_zero_mask = np.abs(mean_field) > 1e-10  
    cv_field[non_zero_mask] = std_field[non_zero_mask] / np.abs(mean_field[non_zero_mask])
    
    return cv_field

def find_max_variance_location(ensemble):
    """Return (x, y, z) index of maximum variance across the ensemble."""
    variances = np.var(ensemble, axis=0)
    max_loc = np.unravel_index(np.argmax(variances), variances.shape)
    return max_loc

def find_max_VC_location(ensemble):
    """Return (x, y, z) index of maximum coefficient-of-variation across ensemble."""
    VC_ens = calculate_cv_field(ensemble)
    max_loc = np.unravel_index(np.argmax(VC_ens), VC_ens.shape)
    return max_loc

def get_observation(true_field, locations):
    """Sample values from a field at provided grid indices and return as column vector."""
    observations = np.array([true_field[loc] for loc in locations])
    return observations.reshape(-1, 1)

def iterative_observation_selection(initial_ensemble, true_field, m, R):
    """Greedy iterative selection of observation points with ETKF update."""
    current_ensemble = initial_ensemble.copy()
    obs_locations = []
    
    for _ in range(m):
        
        new_loc = find_max_variance_location(current_ensemble)
        obs_locations.append(new_loc)
        
        
        observations = get_observation(true_field, obs_locations)
        
        
        current_ensemble = ETKF(current_ensemble, observations, np.array(obs_locations), R)
    
    return current_ensemble, np.array(obs_locations), observations

def etkf_iterative_selection(ensemble, conc_matrix, m, R):
    detectors = []
    
    R=3
    for i in range(m):
        if i == 0:
            detectors.append(find_max_variance_location(ensemble))
        else:
            ana_perturbation = update_ETKF_PERTURBATION(
                ensemble, 
                np.array(detectors), 
                R
            )
            detectors.append(find_max_VC_location(ana_perturbation))
    observations = get_observation(conc_matrix, detectors)
    current_ensemble, mean_analysis = ETKF(ensemble, observations, np.array(detectors), R)
    return detectors, observations, current_ensemble, mean_analysis