"""
Local Ensemble Transform Kalman Filter (LETKF) for 3D concentration fields.
Includes localized analysis, optional Gaussian localization of the observation
operator, and a parallel tiled driver for scalable updates.
"""
import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse import lil_matrix, eye
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import multiprocessing as mp
from functools import partial

def gaussian_localization(distance, localization_radius):
    """Gaussian localization weights for a given distance field and radius."""
    return np.exp(-0.5 * (distance / localization_radius) ** 2)

def _normalize_grid_shape(grid_shape):
    """Normalize provided grid shape to a 3-tuple (nx, ny, nz)."""
    shape = tuple(int(s) for s in grid_shape)
    if len(shape) < 3:
        raise ValueError(f"grid_shape must contain at least 3 dimensions, got: {shape}")
    if len(shape) == 3:
        return shape
    
    return shape[:3]


def calculate_3d_distances(grid_shape, center_point, max_radius=None):
    """Compute Euclidean distances in a 3D grid to a given center point."""
    nx, ny, nz = _normalize_grid_shape(grid_shape)
    cx, cy, cz = center_point
    
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
    )
    
    distance_field = np.sqrt(
        (x_coords - cx) ** 2 + (y_coords - cy) ** 2 + (z_coords - cz) ** 2
    )
    
    if max_radius is not None:
        distance_field[distance_field > max_radius] = np.inf
    
    return distance_field

def localize_observation_operator(H, obs_locations, grid_shape, localization_radius):
    """Apply distance-based Gaussian localization to rows of the H operator."""
    m, n_total = H.shape
    nx, ny, nz = _normalize_grid_shape(grid_shape)
    
    H_localized = np.zeros_like(H)
    
    for i, (ox, oy, oz) in enumerate(obs_locations):
        distances = calculate_3d_distances(grid_shape, (ox, oy, oz), localization_radius)
        
        weights = gaussian_localization(distances, localization_radius)
        weights = weights.flatten()
        
        H_localized[i, :] = H[i, :] * weights
    
    return H_localized

def LETKF_local_analysis(ensemble, observations, obs_locations, R,
                         grid_shape, localization_radius, center_point,
                         use_gaussian_localization: bool = False,
                         inflation: float = 1.0,
                         spread_restoration: str = 'none',
                         rtps_alpha: float = 0.0,
                         rtpp_alpha: float = 0.0,
                         neg_innovation_boost: bool = False,
                         neg_innovation_factor: float = 1.0):
    """Perform localized LETKF analysis around a specified center point.

    Returns local analysis ensemble and mean together with subdomain bounds.
    """
    n, nx, ny, nz = ensemble.shape
    
    x_min = max(0, center_point[0] - localization_radius)
    x_max = min(nx, center_point[0] + localization_radius + 1)
    y_min = max(0, center_point[1] - localization_radius)
    y_max = min(ny, center_point[1] + localization_radius + 1)
    z_min = max(0, center_point[2] - localization_radius)
    z_max = min(nz, center_point[2] + localization_radius + 1)
    
    local_ensemble = ensemble[:, x_min:x_max, y_min:y_max, z_min:z_max]
    local_shape = local_ensemble.shape[1:]
    
    local_ensemble_flat = local_ensemble.reshape(n, -1)
    local_mean = np.mean(local_ensemble_flat, axis=0)
    
    Xp_T = (local_ensemble_flat - local_mean) / np.sqrt(n - 1)
    Xp = Xp_T.T
    
    inside_indices = []
    local_positions = []
    for i, (xi, yi, zi) in enumerate(obs_locations):
        if (x_min <= xi < x_max and y_min <= yi < y_max and z_min <= zi < z_max):
            inside_indices.append(i)
            local_positions.append((xi - x_min, yi - y_min, zi - z_min))
    m_local = len(inside_indices)
    H_local = np.zeros((m_local, local_ensemble_flat.shape[1]))
    for row, (lx, ly, lz) in enumerate(local_positions):
        idx = lx * local_shape[1] * local_shape[2] + ly * local_shape[2] + lz
        H_local[row, idx] = 1
    
    if m_local > 0:
        if observations.ndim == 1:
            obs_flat = observations.reshape(-1)
            observations_local = obs_flat[inside_indices].reshape(m_local, 1)
        else:
            if observations.shape[1] == 1:
                observations_local = observations[inside_indices, :].reshape(m_local, 1)
            else:
                observations_local = observations[inside_indices, 0:1]
    else:
        observations_local = np.zeros((0, 1))
    
    
    def localize_H_for_subdomain(H_local_mat, local_positions_list, local_shape_tuple, radius):
        
        H_loc_out = np.zeros_like(H_local_mat)
        for row_idx, (lx, ly, lz) in enumerate(local_positions_list):
            distances = calculate_3d_distances(local_shape_tuple, (lx, ly, lz), radius)
            weights = gaussian_localization(distances, radius).reshape(-1)
            H_loc_out[row_idx, :] = H_local_mat[row_idx, :] * weights
        return H_loc_out

    
    H_localized = H_local if not use_gaussian_localization else \
        localize_H_for_subdomain(H_local, local_positions, local_shape, localization_radius)

    
    Xp_infl = Xp if inflation == 1.0 else (inflation * Xp)

    
    Yp = H_localized @ Xp_infl  # (m_local, n)

    
    if m_local > 0:
        if np.isscalar(R):
            R_mat = (R + 0.0) * np.eye(m_local)
        else:
            R = np.asarray(R)
            if R.ndim == 1 and R.shape[0] == m_local:
                R_mat = np.diag(R)
            else:
                R_mat = R
        
        R_mat = 0.5 * (R_mat + R_mat.T)
        R_inv = np.linalg.inv(R_mat)
    else:
        R_inv = np.zeros((0, 0))

    
    C = Yp.T @ R_inv @ Yp if m_local > 0 else np.zeros((n, n))
    C = 0.5 * (C + C.T)
    eps = 1e-8
    C += eps * np.eye(n)

    
    w, V = np.linalg.eigh(C)
    inv_sqrt = 1.0 / np.sqrt(1.0 + np.clip(w, 0.0, None))
    T_sym = V @ (np.diag(inv_sqrt) @ V.T)

    
    Xa_prime = Xp_infl @ T_sym  # (s, n)
    analysis_perturbations = (Xa_prime * np.sqrt(n - 1)).T  # (n, s)

    
    if spread_restoration and spread_restoration.lower() != 'none':
        eps_sr = 1e-8
        prior_anomalies = (Xp_infl * np.sqrt(n - 1)).T  # (n, s)
        if spread_restoration.lower() == 'rtps':
            sigma_b = np.std(prior_anomalies, axis=0, ddof=1)
            sigma_a = np.std(analysis_perturbations, axis=0, ddof=1)
            factor = (1.0 - rtps_alpha) + rtps_alpha * (sigma_b / (sigma_a + eps_sr))
            analysis_perturbations = analysis_perturbations * factor.reshape(1, analysis_perturbations.shape[1])
        elif spread_restoration.lower() == 'rtpp':
            analysis_perturbations = (1.0 - rtpp_alpha) * analysis_perturbations + rtpp_alpha * prior_anomalies

    
    if m_local > 0:
        d = observations_local - (H_localized @ local_mean.reshape(-1, 1))  # (m,1)
        if neg_innovation_boost:
            neg_mask = (d < 0).astype(float)
            d = d * (1.0 + neg_mask * (neg_innovation_factor - 1.0))
        
        rhs = Yp.T @ (R_inv @ d)
        A = np.eye(n) + C
        A = 0.5 * (A + A.T) + eps * np.eye(n)
        w_bar = np.linalg.solve(A, rhs)  # (n,1)
        mean_analysis = local_mean.reshape(-1, 1) + Xp_infl @ w_bar
    else:
        mean_analysis = local_mean.reshape(-1, 1)
    
    
    analysis_ensemble_flat = mean_analysis.reshape(-1) + analysis_perturbations
    local_analysis_ensemble = analysis_ensemble_flat.reshape(n, *local_shape)
    local_analysis_mean = mean_analysis.reshape(local_shape)
    
    return local_analysis_ensemble, local_analysis_mean, (x_min, x_max, y_min, y_max, z_min, z_max), m_local

 
import numpy as np
import multiprocessing as mp

def compute_subregion_bounds(i, j, k, sub_size, overlap, nx, ny, nz):
    """Compute subregion bounds (x_start, x_end, y_start, y_end, z_start, z_end)."""
    x_start = i * (sub_size - overlap)
    y_start = j * (sub_size - overlap)
    z_start = k * (sub_size - overlap)

    x_end = min(x_start + sub_size, nx)
    y_end = min(y_start + sub_size, ny)
    z_end = min(z_start + sub_size, nz)

    return x_start, x_end, y_start, y_end, z_start, z_end


def process_subregion(args):
    """Perform LETKF analysis for a single subregion (worker function)."""
    (i, j, k, sub_size, overlap, nx, ny, nz, ensemble, observations,
     obs_locations, R, grid_shape, localization_radius,
     use_gaussian_localization, inflation, spread_restoration, rtps_alpha, rtpp_alpha, neg_innovation_boost, neg_innovation_factor) = args
    
    bounds = compute_subregion_bounds(i, j, k, sub_size, overlap, nx, ny, nz)
    x_start, x_end, y_start, y_end, z_start, z_end = bounds
    
    center_point = ((x_start + x_end) // 2, 
                    (y_start + y_end) // 2, 
                    (z_start + z_end) // 2)

    try:
        local_analysis, local_mean, bounds, m_local = LETKF_local_analysis(
            ensemble, observations, obs_locations, R, grid_shape,
            localization_radius, center_point,
            use_gaussian_localization=use_gaussian_localization,
            inflation=inflation,
            spread_restoration=spread_restoration,
            rtps_alpha=rtps_alpha,
            rtpp_alpha=rtpp_alpha,
            neg_innovation_boost=neg_innovation_boost,
            neg_innovation_factor=neg_innovation_factor
        )
        return (bounds, local_analysis, local_mean, m_local)
    except Exception as e:
        print(f"Error in subregion ({i}, {j}, {k}): {e}")
        return None


def compute_local_weights(bounds, sub_size):
    """Compute a weight field tapering from the subregion center."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    local_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
    local_weight = np.ones(local_shape)

    center_local = (local_shape[0] // 2, local_shape[1] // 2, local_shape[2] // 2)

    for x in range(local_shape[0]):
        for y in range(local_shape[1]):
            for z in range(local_shape[2]):
                dist = np.sqrt((x - center_local[0])**2 + 
                               (y - center_local[1])**2 + 
                               (z - center_local[2])**2)
                local_weight[x, y, z] = gaussian_localization(dist, sub_size / 2)
    return local_weight


def merge_results(results, analysis_ensemble, analysis_mean, weight_field, n, sub_size,
                  background_ensemble, background_mean):
    """Merge all subregion analyses and retain background in uncovered areas."""
    for result in results:
        if result is not None:
            bounds, local_analysis, local_mean, m_local = result
            x_min, x_max, y_min, y_max, z_min, z_max = bounds

            
            if m_local == 0:
                continue
            local_weight = compute_local_weights(bounds, sub_size)

            
            weight_field[x_min:x_max, y_min:y_max, z_min:z_max] += local_weight

            
            for member in range(n):
                analysis_ensemble[member, x_min:x_max, y_min:y_max, z_min:z_max] += (
                    local_analysis[member] * local_weight
                )
            analysis_mean[x_min:x_max, y_min:y_max, z_min:z_max] += (
                local_mean * local_weight
            )

    
    zero_mask = (weight_field == 0)
    safe_weights = weight_field.copy()
    safe_weights[zero_mask] = 1.0
    for member in range(n):
        analysis_ensemble[member] /= safe_weights
    analysis_mean /= safe_weights

    
    if np.any(zero_mask):
        analysis_mean[zero_mask] = background_mean[zero_mask]
        for member in range(n):
            analysis_ensemble[member][zero_mask] = background_ensemble[member][zero_mask]

    return analysis_ensemble, analysis_mean


def LETKF_parallel(ensemble, observations, obs_locations, R,
                  grid_shape, localization_radius, overlap_ratio=0.3,
                  use_gaussian_localization: bool = False,
                  inflation: float = 1.0,
                  spread_restoration: str = 'none',
                  rtps_alpha: float = 0.0,
                  rtpp_alpha: float = 0.0,
                  neg_innovation_boost: bool = False,
                  neg_innovation_factor: float = 1.0):
    """Parallel tiled LETKF over a 3D grid with optional localization and SR."""
    n, nx, ny, nz = ensemble.shape
    
    
    sub_size = int(2 * localization_radius)
    overlap = int(sub_size * overlap_ratio)

    
    nx_sub = max(1, (nx - overlap) // (sub_size - overlap))
    ny_sub = max(1, (ny - overlap) // (sub_size - overlap))
    nz_sub = max(1, (nz - overlap) // (sub_size - overlap))

    
    analysis_ensemble = np.zeros_like(ensemble)
    analysis_mean = np.zeros_like(np.mean(ensemble, axis=0))
    weight_field = np.zeros((nx, ny, nz))

    
    subregion_params = [
        (i, j, k, sub_size, overlap, nx, ny, nz, ensemble, observations,
         obs_locations, R, grid_shape, localization_radius,
         use_gaussian_localization, inflation, spread_restoration, rtps_alpha, rtpp_alpha, neg_innovation_boost, neg_innovation_factor)
        for i in range(nx_sub) for j in range(ny_sub) for k in range(nz_sub)
    ]

    
    with mp.Pool(processes=min(mp.cpu_count(), len(subregion_params))) as pool:
        results = pool.map(process_subregion, subregion_params)

    
    background_mean = np.mean(ensemble, axis=0)
    analysis_ensemble, analysis_mean = merge_results(
        results, analysis_ensemble, analysis_mean, weight_field, n, sub_size,
        background_ensemble=ensemble, background_mean=background_mean
    )

    return analysis_ensemble, analysis_mean


def LETKF(ensemble, observations, obs_locations, R,
         grid_shape=None, localization_radius=5, use_parallel=True,
         use_gaussian_localization: bool = False,
         inflation: float = 1.0,
         spread_restoration: str = 'none',
         rtps_alpha: float = 0.0,
         rtpp_alpha: float = 0.0,
         neg_innovation_boost: bool = False,
         neg_innovation_factor: float = 1.0):
    """Top-level LETKF interface for 3D fields, optionally parallelized."""
    if grid_shape is None:
        grid_shape = ensemble.shape[1:]
    
    if use_parallel:
        return LETKF_parallel(ensemble, observations, obs_locations, R,
                            grid_shape, localization_radius,
                            use_gaussian_localization=use_gaussian_localization,
                            inflation=inflation,
                            spread_restoration=spread_restoration,
                            rtps_alpha=rtps_alpha,
                            rtpp_alpha=rtpp_alpha,
                            neg_innovation_boost=neg_innovation_boost,
                            neg_innovation_factor=neg_innovation_factor)
    else:
        
        center_point = (grid_shape[0] // 2, grid_shape[1] // 2, grid_shape[2] // 2)
        local_analysis, local_mean, _ = LETKF_local_analysis(
            ensemble, observations, obs_locations, R, grid_shape,
            localization_radius, center_point,
            use_gaussian_localization=use_gaussian_localization,
            inflation=inflation,
            spread_restoration=spread_restoration,
            rtps_alpha=rtps_alpha,
            rtpp_alpha=rtpp_alpha,
            neg_innovation_boost=neg_innovation_boost,
            neg_innovation_factor=neg_innovation_factor
        )
        return local_analysis, local_mean

def update_LETKF_PERTURBATION(ensemble, obs_locations, R,
                             grid_shape=None, localization_radius=5,
                             use_gaussian_localization: bool = False,
                             inflation: float = 1.0):
    """Compute analysis perturbations for selection using localized LETKF."""
    if grid_shape is None:
        grid_shape = ensemble.shape[1:]
    
    n, nx, ny, nz = ensemble.shape
    ensemble_flat = ensemble.reshape(n, -1)
    
    
    mean = np.mean(ensemble_flat, axis=0)
    Xp_T = (ensemble_flat - mean) / np.sqrt(n - 1)  # (n, s)
    Xp = Xp_T.T  # (s, n)
    
    
    H = np.zeros((len(obs_locations), nx * ny * nz))
    for i, (xi, yi, zi) in enumerate(obs_locations):
        idx = xi * ny * nz + yi * nz + zi
        H[i, idx] = 1
    
    
    H_localized = H
    if use_gaussian_localization:
        
        grid_shape_eff = (nx, ny, nz)
        try:
            gs_norm = _normalize_grid_shape(grid_shape)
            if np.prod(gs_norm) != nx * ny * nz:
                grid_shape_eff = (nx, ny, nz)
            else:
                grid_shape_eff = tuple(gs_norm)
        except Exception:
            grid_shape_eff = (nx, ny, nz)
        H_localized = localize_observation_operator(H, obs_locations, grid_shape_eff, localization_radius)
    
    
    Xp_infl = Xp if inflation == 1.0 else (inflation * Xp)
    HX = H_localized @ Xp_infl  # (m, n)
    
    m = len(obs_locations)
    if np.isscalar(R):
        R_mat = (R + 0.0) * np.eye(m)
    else:
        R = np.asarray(R)
        if R.ndim == 1 and R.shape[0] == m:
            R_mat = np.diag(R)
        else:
            R_mat = R
    R_mat = 0.5 * (R_mat + R_mat.T)
    R_inv = np.linalg.inv(R_mat)
    C = HX.T @ R_inv @ HX  # (n, n)
    C = 0.5 * (C + C.T)
    C += 1e-8 * np.eye(n)
    
    
    w, V = np.linalg.eigh(C)
    T = V @ (np.diag(1.0 / np.sqrt(1.0 + np.clip(w, 0.0, None))) @ V.T)
    
    
    Xa_prime = Xp_infl @ T  # (s, n)
    analysis_perturbations = (Xa_prime * np.sqrt(n - 1)).T  # (n, s)
    
    return analysis_perturbations.reshape(n, nx, ny, nz)

def letkf_posterior_variance_field(ensemble, obs_locations, R,
                                   grid_shape=None, localization_radius=5,
                                   use_gaussian_localization: bool = False,
                                   inflation: float = 1.0):
    """Estimate posterior variance field from LETKF analysis perturbations."""
    
    ana_pert = update_LETKF_PERTURBATION(
        ensemble,
        obs_locations,
        R,
        grid_shape=grid_shape,
        localization_radius=localization_radius,
        use_gaussian_localization=use_gaussian_localization,
        inflation=inflation
    )  # (n, x, y, z)
    
    post_var = np.var(ana_pert, axis=0, ddof=1)
    return post_var


def analyze_xy_concentration(conc_field):
    """Analyze XY-plane distribution features of a 3D concentration field.

    Returns the number of peaks and a qualitative concentration description,
    and produces diagnostic plots.
    """
    
    xy_mean = np.mean(conc_field, axis=2)
    
    
    xy_smoothed = gaussian_filter(xy_mean, sigma=1)
    
    flattened = xy_smoothed.flatten()
    peaks, _ = find_peaks(flattened, prominence=np.std(flattened)/2)
    num_peaks = len(peaks)
    
    from scipy.stats import kurtosis
    kurt = kurtosis(flattened)
    if kurt > 3:
        concentration = "concentrated"
    else:
        concentration = "dispersed"
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(xy_mean, cmap='viridis')
    plt.colorbar()
    plt.title('XY-plane mean concentration')
    
    plt.subplot(132)
    plt.imshow(xy_smoothed, cmap='viridis')
    plt.colorbar()
    plt.title('Smoothed concentration')
    
    plt.subplot(133)
    hist, bins = np.histogram(flattened, bins=50)
    plt.plot(bins[:-1], hist)
    plt.title('Concentration histogram')
    
    plt.tight_layout()
    plt.show()
    return num_peaks, concentration

def calculate_cv_field(array):
    """Compute coefficient-of-variation field across ensembles (safe at zero mean)."""
    
    mean_field = np.mean(array, axis=0)
    
    std_field = np.std(array, axis=0, ddof=1)  
    
    cv_field = np.zeros_like(mean_field)
    
    non_zero_mask = np.abs(mean_field) > 1e-10 
    cv_field[non_zero_mask] = std_field[non_zero_mask] / np.abs(mean_field[non_zero_mask])
    
    return cv_field

def find_max_variance_location(ensemble):
    """Return (x, y, z) location of maximum variance across the ensemble."""
    variances = np.var(ensemble, axis=0)
    max_loc = np.unravel_index(np.argmax(variances), variances.shape)
    return max_loc

def find_max_VC_location(ensemble):
    """Return (x, y, z) location of maximum coefficient of variation."""
    VC_ens = calculate_cv_field(ensemble)
    max_loc = np.unravel_index(np.argmax(VC_ens), VC_ens.shape)
    return max_loc

def get_observation(true_field, locations):
    """Sample values from a field at provided grid indices and return column vector."""
    observations = np.array([true_field[loc] for loc in locations])
    return observations.reshape(-1, 1)

def iterative_observation_selection(initial_ensemble, true_field, m, R, 
                                 use_letkf=True, localization_radius=5):
    """Greedy iterative selection of observation points with LETKF/ETKF update."""
    current_ensemble = initial_ensemble.copy()
    obs_locations = []
    
    for _ in range(m):
        new_loc = find_max_variance_location(current_ensemble)
        obs_locations.append(new_loc)
        
        observations = get_observation(true_field, obs_locations)
        
        if use_letkf:
            current_ensemble, _ = LETKF(current_ensemble, observations, np.array(obs_locations), R,
                                      localization_radius=localization_radius)
        else:
            from .ETKF import ETKF
            current_ensemble, _ = ETKF(current_ensemble, observations, np.array(obs_locations), R)
    
    return current_ensemble, np.array(obs_locations), observations

def letkf_iterative_selection(ensemble, conc_matrix, m, R, localization_radius=5):
    """Iterative detector selection using LETKF-based perturbation updates."""
    detectors = []
    
    for i in range(m):
        if i == 0:
            detectors.append(find_max_variance_location(ensemble))
        else:
            ana_perturbation = update_LETKF_PERTURBATION(
                ensemble, 
                np.array(detectors), 
                R,
                localization_radius=localization_radius
            )
            detectors.append(find_max_VC_location(ana_perturbation))
    
    observations = get_observation(conc_matrix, detectors)
    current_ensemble, mean_analysis = LETKF(ensemble, observations, np.array(detectors), R,
                                           localization_radius=localization_radius)
    return detectors, observations, current_ensemble, mean_analysis
