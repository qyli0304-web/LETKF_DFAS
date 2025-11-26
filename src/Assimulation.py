"""
High-level assimilation driver orchestrating ETKF/LETKF analysis, adaptive
area weighting, detector selection, and I/O with model and observations.
"""
from src.ETKF import *
from src.LETKF import *
from src.polyphemus_config import ConfigParser
from src.utils import *
import sys
from src.areamask import generate_country_mask_fast
from src.LocalKalman_gaussweigh import LocalKalmanFilter3D
from src.ETKF import update_ETKF_PERTURBATION, ETKF
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AssimilationDriver:
    """Driver to configure and run the data assimilation workflow."""
    def __init__(self, config_file):
        self.assimlu_config = ConfigParser()
        self.assimlu_config.parse(config_file)
        self.influence_range = (4, 4, 2)  
        self.Nensemble = None
        self.Ndetection = int(self.assimlu_config.get("enkf")["Ndetection"])
        self.Nstate = None
        self.domain_size = None

        self.use_letkf = self.assimlu_config.get("flags")["use_letkf"].lower() == "true"
        self.use_letkf_for_selection = (
            self.assimlu_config.get("flags")["use_letkf_for_selection"].lower()
            == "true"
        )
        if self.use_letkf_for_selection or self.use_letkf:
            self.localization_radius = int(
                self.assimlu_config.get("letkf")["localization_radius"]
            )
            self.overlap_ratio = float(
                self.assimlu_config.get("letkf")["overlap_ratio"]
            )
            self.use_parallel = (
                self.assimlu_config.get("letkf")["use_parallel"].lower() == "true"
            )
            self.use_gaussian_localization = (
                self.assimlu_config.get("letkf")["use_gaussian_localization"].lower()
                == "true"
            )
            try:
                self.inflation = float(self.assimlu_config.get("letkf")["inflation"])
            except Exception:
                self.inflation = 1.0
            try:
                self.letkf_spread_restoration = self.assimlu_config.get("letkf")[
                    "spread_restoration"
                ]
            except Exception:
                self.letkf_spread_restoration = "none"
            try:
                self.letkf_rtps_alpha = float(
                    self.assimlu_config.get("letkf")["rtps_alpha"]
                )
            except Exception:
                self.letkf_rtps_alpha = 0.0
            try:
                self.letkf_rtpp_alpha = float(
                    self.assimlu_config.get("letkf")["rtpp_alpha"]
                )
            except Exception:
                self.letkf_rtpp_alpha = 0.0
        try:
            self.etkf_prior_inflation = float(
                self.assimlu_config.get("etkf")["prior_inflation"]
            )
        except Exception:
            self.etkf_prior_inflation = 1.0
        try:
            self.etkf_posterior_inflation = float(
                self.assimlu_config.get("etkf")["posterior_inflation"]
            )
        except Exception:
            self.etkf_posterior_inflation = 1.0
        try:
            self.spread_restoration = self.assimlu_config.get("enkf")[
                "spread_restoration"
            ]
        except Exception:
            self.spread_restoration = "none"
        try:
            self.rtps_alpha = float(self.assimlu_config.get("enkf")["rtps_alpha"])
        except Exception:
            self.rtps_alpha = 0.0
        try:
            self.rtpp_alpha = float(self.assimlu_config.get("enkf")["rtpp_alpha"])
        except Exception:
            self.rtpp_alpha = 0.0
        try:
            self.positivity_correction = (
                self.assimlu_config.get("flags")["positivity_correction"].lower()
                == "true"
            )
        except Exception:
            self.positivity_correction = True
        try:
            self.positivity_method = self.assimlu_config.get("enkf")[
                "positivity_method"
            ]
        except Exception:
            self.positivity_method = "clip"  # options: none, clip, mean_preserve, log
        try:
            self.log_eps = float(self.assimlu_config.get("enkf")["log_eps"])
        except Exception:
            self.log_eps = 1e-8
        try:
            self.neg_innovation_boost = (
                self.assimlu_config.get("enkf")["neg_innovation_boost"].lower()
                == "true"
            )
        except Exception:
            self.neg_innovation_boost = False
        try:
            self.neg_innovation_factor = float(
                self.assimlu_config.get("enkf")["neg_innovation_factor"]
            )
        except Exception:
            self.neg_innovation_factor = 1.0
        self.ensemble_conc = None
        self.forward_dir = None
        self.backward_dir = None
        self.release_name = None
        self.inverse_iteration = (
            self.assimlu_config.get("flags")["inverse_iteration"].lower() == "true"
        )
        self.release_recorrect = (
            self.assimlu_config.get("flags")["release_recorrect"].lower() == "true"
        )
        self.area_limit = (
            self.assimlu_config.get("flags")["area_limit"].lower() == "true"
        )
        self.iteration_assimulation = (
            self.assimlu_config.get("flags")["iteration_assimulation"].lower() == "true"
        )
        self.key_area_constraction = (
            self.assimlu_config.get("flags")["key_area_constraction"].lower() == "true"
        )
        self.height_limit = (
            self.assimlu_config.get("flags")["height_restriction"].lower() == "true"
        )
        self.flag_dump_detection = (
            self.assimlu_config.get("flags")["dump_detection"].lower() == "true"
        )
        self.flag_dump_keyarea = (
            self.assimlu_config.get("flags")["dump_keyarea_message"].lower() == "true"
        )
        self.flag_PoBackinPlmue = (
            self.assimlu_config.get("flags")["PoBackinPlmue"].lower() == "true"
        )
        try:
            self.flag_print_sensitive_windows = (
                self.assimlu_config.get("flags")["print_sensitive_windows"].lower()
                == "true"
            )
        except Exception:
            self.flag_print_sensitive_windows = False
        
        try:
            self.flag_dump_p_field = (
                self.assimlu_config.get("flags")["dump_p_field"].lower() == "true"
            )
        except Exception:
            self.flag_dump_p_field = False

        self.flag_fixed_station = (
            self.assimlu_config.get("flags")["fixed_station"].lower() == "true"
        )
        if self.flag_fixed_station:
            self.info_station_dir = Path(
                self.assimlu_config.get("path")["info_station_dir"]
            ).resolve()
            self.read_station = True
        if self.flag_dump_keyarea:
            self.key_message_dir = Path(
                self.assimlu_config.get("path")["key_message_dir"]
            )

        if self.flag_dump_detection:
            self.detection_dir = Path(self.assimlu_config.get("path")["detection_dir"])

        try:
            adaptive_cfg = self.assimlu_config.get("Adaptive")
            self.adaptive_enable = adaptive_cfg["enable"].lower() == "true"
            self.gukou_hours = (
                float(adaptive_cfg["gukou_hours"])
                if "gukou_hours" in adaptive_cfg
                else 12.0
            )
            self.adaptive_weight_sensitive = (
                float(adaptive_cfg["sensitive_weight"])
                if "sensitive_weight" in adaptive_cfg
                else 1.0
            )
            self.adaptive_weight_nonsensitive = (
                float(adaptive_cfg["nonsensitive_weight"])
                if "nonsensitive_weight" in adaptive_cfg
                else 0.3
            )
            self.adaptive_weight_expired = (
                float(adaptive_cfg["expired_weight"])
                if "expired_weight" in adaptive_cfg
                else 0.0
            )
            self.adaptive_method = (
                int(adaptive_cfg["method"]) if "method" in adaptive_cfg else 1
            )
        except Exception:
            self.adaptive_enable = False
            self.gukou_hours = 12.0
            self.adaptive_weight_sensitive = 1.0
            self.adaptive_weight_nonsensitive = 0.3
            self.adaptive_weight_expired = 0.0
            self.adaptive_method = 1

        self.forward_scheme = self.assimlu_config.get("enkf")["forward_scheme"]
        self.backward_scheme = self.assimlu_config.get("enkf")["backward_scheme"]

        self.assimulation_step_list = None

        self.current_time_step = None
        self.last_assimlation_step = None

        if self.iteration_assimulation:
            step_begin = int(self.assimlu_config.get("enkf")["assimilation_start"])
            step_end = int(self.assimlu_config.get("enkf")["assimilation_end"])
            step_interval = int(
                self.assimlu_config.get("enkf")["assimilation_hour_delta"]
            )
            self.assimulation_step_list = list(
                range(step_begin, step_end + 1, step_interval)
            )
        else:
            self.assimulation_step_list = [
                int(step)
                for step in self.assimlu_config.get("enkf")[
                    "assimulation_step_list"
                ].split(",")
            ]

        if self.area_limit:
            self.limited_area_name = self.assimlu_config.get("enkf")[
                "limited_area_name"
            ]

        self.backward_weight = float(
            self.assimlu_config.get("enkf")["backward_weight"]
        )
        self.L = 0.1 * (1 - self.backward_weight)

        if self.key_area_constraction:
            self.flag_population = (
                self.assimlu_config.get("flags")["population_weight"].lower() == "true"
            )
            self.flag_conc = (
                self.assimlu_config.get("flags")["conc_weight"].lower() == "true"
            )

        self.area_mask = None
        self.model = None
        self.obs_manager = None
        self.origin_result_size = None
        self.current_result_size = None
        self.backward_result_size = None
        self.detector_list = None
        self.tran_geo_detector_list = None
        self.ana_conc = None
        self.ensemble_ana_conc = None

        if self.height_limit:
            self.limit_level = int(self.assimlu_config.get("enkf")["limit_level"])

        self.detector_values = None

        level_path = self.assimlu_config.get("path")["level_path"]
        levels = np.loadtxt(level_path).tolist()
        self.levels = [(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)]

    def initialize(self, model, obs_manager):
        """Initialize model runs, key areas, and area mask prior to assimilation."""
        self.model = model
        self.model.initialize()
        self.current_time_step = 0
        self.obs_manager = obs_manager
        self.release_name = self.model.get_release_name()
        self.forward_name = self.model.get_forward_name()
        self.Nensemble = self.model.get_Nensemble()
        self.forward_dir = self.model.get_forward_dir()
        self.backward_dir = self.model.get_backward_dir()
        self.domain_size = self.model.get_domain_size()
        self.origin_result_size = self.model.get_origin_result_size()
        self.backward_result_size = self.origin_result_size
        self.current_result_size = self.model.get_current_result_size()

        model.run_forward()

        self.model.initialize_keyarea()
        self.model.update_keyarealist()
        limit_value = self.model.get_limited_value()
        with open("limited_value.txt", "w") as f:
            f.write(f"limited value is {limit_value}")
        self.model.update_output_keyarealist()
        if self.flag_dump_keyarea:
            self.dump_keyarea_message()
            self.dump_conclist()
        if (
            self.key_area_constraction or self.adaptive_enable
        ) and not self.flag_fixed_station:
            model.run_backward()
        if self.area_limit:
            mask, _, _ = generate_country_mask_fast(
                self.domain_size[0],
                self.domain_size[1],
                self.origin_result_size[0],
                self.origin_result_size[1],
                self.domain_size[2],
                self.domain_size[3],
                country_name=self.limited_area_name,
            )
            self.area_mask = mask.copy().T

    def _compute_adaptive_area_weights(self):
        """Compute adaptive weights for key areas based on sensitive windows.

        Returns a tuple (area_weights, has_sensitive_area, earlier_than_all).
        """
        area_weights = {}
        keyarea_list = self.model.get_keyarea_list()
        assimilation_dt = self.model.simulation_date_min + timedelta(
            hours=(self.current_time_step + 1) * self.model.sim_save_delta_t
        )
        window_starts = []
        has_sensitive = False

        for area in keyarea_list:
            name = area.get_areaname()
            arrive_times = area.get_arrivetime()
            if arrive_times is None or len(arrive_times) == 0:
                area_weights[name] = self.adaptive_weight_nonsensitive
                continue
            sorted_times = sorted(arrive_times)
            median_idx = len(sorted_times) // 2
            arrive_dt = sorted_times[median_idx]
            window_start = arrive_dt - timedelta(hours=self.gukou_hours)
            window_starts.append(window_start)
            if self.flag_print_sensitive_windows:
                try:
                    sim0 = self.model.simulation_date_min
                    ws_h = (window_start - sim0).total_seconds() / 3600.0
                    mid_h = (arrive_dt - sim0).total_seconds() / 3600.0
                    print(
                        f"[SensitiveWindow] {name}: {window_start.strftime('%Y-%m-%d_%H-%M-%S')} ~ {arrive_dt.strftime('%Y-%m-%d_%H-%M-%S')} | hours: [{ws_h:.0f}, {mid_h:.0f}] | width={self.gukou_hours}h"
                    )
                except Exception:
                    pass
            if window_start <= assimilation_dt <= arrive_dt:
                area_weights[name] = self.adaptive_weight_sensitive
                has_sensitive = True
            elif assimilation_dt < window_start:
                area_weights[name] = self.adaptive_weight_nonsensitive
            else:  
                area_weights[name] = self.adaptive_weight_expired

        earlier_than_all = False
        if len(window_starts) > 0:
            earliest_start = min(window_starts)
            earlier_than_all = assimilation_dt < earliest_start
        return area_weights, has_sensitive, earlier_than_all

    def set_step(self, step):
        self.current_time_step = step

    def update_step(self, step):
        self.model.set_step(step)
        self.set_step(step)

    def initial_state(self):
        """Load and prepare the ensemble concentration field for this step."""
        if self.release_recorrect:
            print(
                "the release recorrect feature is not currently supported.",
                file=sys.stderr,
            )
            
            sys.exit(1)  
        else:
            self.Nstate = (
                self.current_result_size[0]
                * self.current_result_size[1]
                * self.current_result_size[2]
            )
            if self.last_assimlation_step == None:
                this_step = self.current_time_step
            else:
                this_step = self.current_time_step - self.last_assimlation_step - 1
            self.ensemble_conc = load_ensemble(
                self.forward_dir,
                n_ensemble=self.Nensemble,
                n_step=this_step,
                sizes=self.current_result_size,
                filename=self.forward_name + "_" + self.release_name + "_air.bin",
            )
            

    def get_ensemble_conc(self):
        if self.ensemble_conc is None:
            print(
                "The ensemble concentration field has not been initialized.",
                file=sys.stderr,
            )
            sys.exit(1)
        return self.ensemble_conc

    def cal_forward_P_field(self, ensemble_conc):
        """Compute forward uncertainty field (variance/CV/mixed) from ensemble."""
        if self.height_limit:
            need_field = ensemble_conc[:, :, :, : self.limit_level + 1].copy()
        else:
            need_field = ensemble_conc.copy()
        if self.area_limit:
            need_field = need_field * self.area_mask[None, :, :, None]
        match self.forward_scheme:
            case "variance":
                forward_P = np.var(need_field, axis=0)
            case "CV":
                forward_P = calculate_cv_field(need_field)
            case "mixed":
                forward_P = (
                    np.var(need_field, axis=0) + calculate_cv_field(need_field)
                ) / 2
                

        return forward_P

    def cal_backward_P_field(self, area_weights=None):
        """Aggregate backward fields from inverse runs, with optional area weights."""
        keyarea_list = self.model.get_keyarealist()
        name_list = []
        for keyarea in keyarea_list:
            name_list.append(keyarea.get_areaname())
        final_part_name = "_" + self.release_name + "_air.bin"
        back_conc_dict = load_inverse_ensemble(
            self.backward_dir,
            n_ensemble=self.Nensemble,
            n_step=self.current_time_step,
            sizes=self.backward_result_size,
            city_list=name_list,
            filename=final_part_name,
        )
        if self.backward_scheme == "hotspot":
            need_field = np.zeros(self.backward_result_size[0:3])
            for i, name in enumerate(name_list):
                for j in range(self.Nensemble):
                    this_field = back_conc_dict[name][j, :, :, :].copy()
                    if self.flag_population:
                        this_field = this_field * keyarea_list[i].get_population()
                    if self.flag_conc:
                        this_field = this_field * keyarea_list[i].get_meanconc()[j]
                    
                    if area_weights is not None:
                        w = area_weights.get(name, 1.0)
                        this_field = this_field * w
                    need_field += this_field
            if self.height_limit:
                need_field = need_field[:, :, : self.limit_level + 1]
            if self.area_limit:
                need_field = need_field * self.area_mask[:, :, None]

            return need_field
        elif self.backward_scheme == "entropy":
            print("the entropy feature is not currently supported.", file=sys.stderr)
            sys.exit(1)  

    def find_max_loaction(self, array):
        """Return the index of the maximum value in a 3D array."""
        max_index = np.unravel_index(np.argmax(array), array.shape)
        return max_index

    def find_max_loaction_with_value(self, P_field):
        mask = np.ones_like(P_field, dtype=bool)
        border_width = 1
        mask[
            border_width:-border_width,
            border_width:-border_width,
            border_width:-border_width,
        ] = False

        temp_field = P_field.copy()
        temp_field[mask] = -np.inf

        flat_idx = np.argmax(temp_field)
        max_loc = np.unravel_index(flat_idx, temp_field.shape)
        max_val = temp_field[max_loc]

        return max_loc, max_val

    def generate_3d_mask(self, matrix_dim, relative_coords, influence_range, sigma=1.0):
        """Generate a 3D Gaussian-influence mask around given coordinates."""
        nx, ny, nz = matrix_dim
        n1, n2, n3 = influence_range
        mask = np.zeros((nx, ny, nz))

        for xc, yc, zc in relative_coords:
            x_min = max(0, xc - n1)
            x_max = min(nx, xc + n1 + 1)
            y_min = max(0, yc - n2)
            y_max = min(ny, yc + n2 + 1)
            z_min = max(0, zc - n3)
            z_max = min(nz, zc + n3 + 1)

            x = np.arange(x_min, x_max)
            y = np.arange(y_min, y_max)
            z = np.arange(z_min, z_max)
            xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
            dx = (xx - xc) / n1
            dy = (yy - yc) / n2
            dz = (zz - zc) / n3
            distance_sq = dx**2 + dy**2 + dz**2
            gaussian_weights = np.exp(-distance_sq / (2 * sigma**2))

            mask[x_min:x_max, y_min:y_max, z_min:z_max] = np.maximum(
                mask[x_min:x_max, y_min:y_max, z_min:z_max], gaussian_weights
            )

        return mask

    def update_detectors(self):
        self.detector_list = []

        P_forward = self.cal_forward_P_field(self.ensemble_conc)
        P_backward = None
        if self.adaptive_enable:
            area_weights, has_sensitive, earlier_than_all = (
                self._compute_adaptive_area_weights()
            )
            assimilation_dt = self.model.simulation_date_min + timedelta(
                hours=(self.current_time_step + 1) * self.model.sim_save_delta_t
            )
            if getattr(self, "adaptive_method", 1) == 2:
                if earlier_than_all:
                    keyarea_list = self.model.get_keyarea_list()
                    equal_weights = {ka.get_areaname(): 1.0 for ka in keyarea_list}
                    P_backward = self.cal_backward_P_field(area_weights=equal_weights)
                    method_str = "Forward+Backward (equal weights)"
                else:
                    P_backward = self.cal_backward_P_field(area_weights=area_weights)
                    method_str = "Forward+Backward (weighted)"
                print(
                    f"[Adaptive] Assimilation time {assimilation_dt.strftime('%Y-%m-%d_%H-%M-%S')} | In sensitive window: {has_sensitive} | Selection scheme: {method_str} | Method: 2"
                )
            else:
                method_str = "Forward only" if earlier_than_all else "Forward+Backward (weighted)"
                print(
                    f"[Adaptive] Assimilation time {assimilation_dt.strftime('%Y-%m-%d_%H-%M-%S')} | In sensitive window: {has_sensitive} | Selection scheme: {method_str} | Method: 1"
                )
                if earlier_than_all:
                    P_backward = None
                else:
                    P_backward = self.cal_backward_P_field(area_weights=area_weights)
        else:
            P_backward = (
                self.cal_backward_P_field() if self.key_area_constraction else None
            )
            method_str = "Forward+Backward" if self.key_area_constraction else "Forward only"
            print(f"[Adaptive] Disabled | Selection scheme: {method_str}")
        max_forw = np.max(P_forward)  
        
        

        
        
        
        P_backward_original = P_backward.copy() if P_backward is not None else None
        
        if (self.key_area_constraction and not self.adaptive_enable) or (
            self.adaptive_enable and P_backward is not None
        ):
            if self.flag_PoBackinPlmue:
                print("[Adaptive] Backward constrained by forward coverage enabled (PoBackinPlmue)")
                P_backward[P_forward <= self.L] = 0  # backward field limited by forward coverage
                P_field = (1 - self.backward_weight) * normalize_data(
                    P_forward
                ) + self.backward_weight * normalize_data(P_backward)
            else:
                P_field = (1 - self.backward_weight) * normalize_data(
                    P_forward
                ) + self.backward_weight * normalize_data(P_backward)
        else:
            P_field = normalize_data(P_forward.copy())
        
        self.dump_p_field(P_forward, P_backward_original, P_field)

        for i in range(self.Ndetection):
            max_loc, max_val = self.find_max_loaction_with_value(P_field)
            print(max_val)
            if max_val < 1e-10:  # Threshold for meaningful concentration
                break

            self.detector_list.append(max_loc)

            if self.use_letkf_for_selection:
                post_var = letkf_posterior_variance_field(
                    self.ensemble_conc,
                    self.detector_list,
                    R=self.obs_manager.get_obs_var(),
                    grid_shape=self.current_result_size,
                    localization_radius=self.localization_radius,
                    use_gaussian_localization=self.use_gaussian_localization,
                    inflation=self.inflation,
                )
                if self.height_limit:
                    post_var = post_var[:, :, : self.limit_level + 1]
                if self.area_limit and self.area_mask is not None:
                    post_var = post_var * self.area_mask[:, :, None]
                P_forward_TEMP = post_var / max_forw
            else:
                ana_ensemble = update_ETKF_PERTURBATION(
                    self.ensemble_conc,
                    self.detector_list,
                    R=self.obs_manager.get_obs_var(),
                    prior_inflation=self.etkf_prior_inflation,
                )
                P_forward_TEMP = self.cal_forward_P_field(ana_ensemble) / max_forw

            if (self.key_area_constraction and not self.adaptive_enable) or (
                self.adaptive_enable and P_backward is not None
            ):

                back_filter = LocalKalmanFilter3D(
                    P_backward,
                    global_variance=5.0,
                    process_noise=0.1,
                    measurement_noise=1.0,
                    local_window=(8, 8, 3),
                    decay_radius=4,
                    weight_threshold=1e-3,
                )
                for _ in range(3):
                    back_filter.update(
                        self.detector_list, np.zeros(len(self.detector_list))
                    )
                P_backward_TEMP = back_filter.get_concentration()
                
                P_field = (
                    1 - self.backward_weight
                ) * P_forward_TEMP + self.backward_weight * P_backward_TEMP
            else:
                P_field = P_forward_TEMP

            mask = self.generate_3d_mask(
                P_field.shape, self.detector_list, self.influence_range
            )
            P_field = P_field * (1 - mask)
        
        

    

    def assimilation(self):
        """
        Execute one full assimilation cycle for the current time step.
        """
        if self.detector_list is None:
            print("The detector list has not been initialized.", file=sys.stderr)
            sys.exit(1)
        if not self.flag_fixed_station:
            self.tran_geo_detector_list = []

            for x, y, z in self.detector_list:
                lon, lat = cordinate2geo(x, y, self.domain_size)
                heigh = self.levels[z]
                self.tran_geo_detector_list.append((lon, lat, heigh))
        self.obs_manager.cal_obsvalue(
            self.tran_geo_detector_list, self.current_time_step
        )
        dv = self.obs_manager.get_obsvalue()
        self.detector_values = np.atleast_1d(dv)

        use_log = self.positivity_correction and (
            self.positivity_method.lower() == "log"
        )
        if use_log:
            ens_in = np.log(self.ensemble_conc + self.log_eps)
            obs_in = np.log(
                np.asarray(self.detector_values).reshape(-1, 1) + self.log_eps
            )
            R_in = self.obs_manager.get_obs_var()
        else:
            ens_in = self.ensemble_conc
            obs_in = self.detector_values
            R_in = self.obs_manager.get_obs_var()

        if self.use_letkf:
            print(f"Using LETKF with localization radius: {self.localization_radius}")
            ana_ensemble, ana_mean = LETKF(
                ens_in,
                obs_in,
                self.detector_list,
                R=R_in,
                grid_shape=self.current_result_size,
                localization_radius=self.localization_radius,
                use_parallel=self.use_parallel,
                use_gaussian_localization=self.use_gaussian_localization,
                inflation=self.inflation,
                spread_restoration=(
                    self.letkf_spread_restoration
                    if self.spread_restoration == "none"
                    else self.spread_restoration
                ),
                rtps_alpha=(
                    self.rtps_alpha
                    if self.letkf_spread_restoration == "none"
                    else self.letkf_rtps_alpha
                ),
                rtpp_alpha=(
                    self.rtpp_alpha
                    if self.letkf_spread_restoration == "none"
                    else self.letkf_rtpp_alpha
                ),
                neg_innovation_boost=self.neg_innovation_boost,
                neg_innovation_factor=self.neg_innovation_factor,
            )
        else:
            print("Using ETKF")
            ana_ensemble, ana_mean = ETKF(
                ens_in,
                obs_in,
                self.detector_list,
                R=R_in,
                prior_inflation=self.etkf_prior_inflation,
                posterior_inflation=self.etkf_posterior_inflation,
                spread_restoration=self.spread_restoration,
                rtps_alpha=self.rtps_alpha,
                rtpp_alpha=self.rtpp_alpha,
                neg_innovation_boost=self.neg_innovation_boost,
                neg_innovation_factor=self.neg_innovation_factor,
            )

        if use_log:
            ana_ensemble = np.exp(ana_ensemble) - self.log_eps
            ana_mean = np.exp(ana_mean) - self.log_eps

        if self.positivity_correction and self.positivity_method.lower() in [
            "clip",
            "mean_preserve",
        ]:
            ana_ensemble = np.maximum(ana_ensemble, 0.0)
            ana_mean = np.maximum(ana_mean, 0.0)
            if self.positivity_method.lower() == "mean_preserve":
                eps = 1e-12
                curr_mean = np.mean(ana_ensemble, axis=0)
                target_mean = ana_mean
                scale = np.ones_like(target_mean)
                need_mask = (
                    (target_mean > 0)
                    & (curr_mean > 0)
                    & (np.any(ana_ensemble > 0, axis=0))
                )
                scale[need_mask] = target_mean[need_mask] / (curr_mean[need_mask] + eps)
                scale_full = scale[np.newaxis, ...]
                pos_mask = ana_ensemble > 0
                ana_ensemble = np.where(
                    pos_mask, ana_ensemble * scale_full, ana_ensemble
                )
                
                zero_mean_mask = (curr_mean <= eps) & (target_mean > 0)
                if np.any(zero_mean_mask):
                    assign_val = (
                        target_mean[zero_mean_mask] / float(self.Nensemble)
                    ).astype(ana_ensemble.dtype)
                    x_idx, y_idx, z_idx = np.where(zero_mean_mask)
                    if x_idx.size > 0:
                        ana_ensemble[:, x_idx, y_idx, z_idx] = assign_val[np.newaxis, :]
                ana_mean = np.mean(ana_ensemble, axis=0)
        self.ana_conc = ana_mean
        self.ensemble_ana_conc = ana_ensemble
        

    def get_simulation_and_assimilation(self):
        simulation = []
        assimilation = []
        for x, y, z in self.detector_list:
            simu_value = self.ensemble_conc[:, x, y, z]
            simulation.append(simu_value)
            assimilation.append(self.ana_conc[x, y, z])
        return simulation, assimilation

    def get_detector_value(self):
        return self.detector_values

    def dump_ana_result(self):
        needpath = self.model.get_dump_path()
        dumppath = needpath / str(self.current_time_step)
        dumppath.mkdir(parents=True, exist_ok=True)

        dump_file_path = dumppath / "ana_mean.bin"
        write_data(dump_file_path, self.ana_conc)

        ana_ensemble_path = dumppath / "ana_ensemble"
        ana_ensemble_path.mkdir(parents=True, exist_ok=True)

        
        for i in range(self.Nensemble):
            ensemble_file_path = ana_ensemble_path / f"ensemble_{i:02d}.bin"
            write_data(ensemble_file_path, self.ensemble_ana_conc[i])

        
        stats_path = dumppath / "ana_stats"
        stats_path.mkdir(parents=True, exist_ok=True)

        
        variance_field = np.var(self.ensemble_conc, axis=0)
        variance_file_path = stats_path / "ana_variance.bin"
        write_data(variance_file_path, variance_field)

        
        std_field = np.std(self.ensemble_conc, axis=0)
        std_file_path = stats_path / "ana_std.bin"
        write_data(std_file_path, std_field)

        
        cv_field = calculate_cv_field(self.ensemble_conc)
        cv_file_path = stats_path / "ana_cv.bin"
        write_data(cv_file_path, cv_field)

        
        stats_info = {
            "time_step": int(self.current_time_step),
            "ensemble_size": int(self.Nensemble),
            "domain_size": [int(x) for x in self.current_result_size],
            "detector_count": int(len(self.detector_list)),
            "detector_locations": [[int(x) for x in loc] for loc in self.detector_list],
            "detector_values": (
                [float(x) for x in self.detector_values]
                if hasattr(self.detector_values, "__iter__")
                else float(self.detector_values)
            ),
            "analysis_mean_min": float(np.min(self.ana_conc)),
            "analysis_mean_max": float(np.max(self.ana_conc)),
            "analysis_mean_mean": float(np.mean(self.ana_conc)),
            "analysis_ensemble_variance_min": float(np.min(variance_field)),
            "analysis_ensemble_variance_max": float(np.max(variance_field)),
            "analysis_ensemble_variance_mean": float(np.mean(variance_field)),
        }

        stats_json_path = stats_path / "ana_stats.json"
        with open(stats_json_path, "w", encoding="utf-8") as f:
            json.dump(stats_info, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

        
        readme_path = dumppath / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# Restart Folder - Time step {self.current_time_step}\n\n")
            f.write(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Folder Structure\n\n")
            f.write("```\n")
            f.write(f"{dumppath.name}/\n")
            f.write("├── ana_mean.bin              # Analysis mean field\n")
            f.write("├── ana_ensemble/             # Analysis ensemble folder\n")
            f.write("│   ├── ensemble_00.bin       # Analysis field of ensemble member 0\n")
            f.write("│   ├── ensemble_01.bin       # Analysis field of ensemble member 1\n")
            f.write("│   ├── ...                    # ...\n")
            f.write(
                "│   └── ensemble_{self.Nensemble-1:02d}.bin  # Analysis field of member {self.Nensemble-1}\n"
            )
            f.write("├── ana_stats/                # Analysis statistics folder\n")
            f.write("│   ├── ana_variance.bin      # Ensemble variance field\n")
            f.write("│   ├── ana_std.bin           # Ensemble standard deviation field\n")
            f.write("│   ├── ana_cv.bin            # Ensemble coefficient of variation field\n")
            f.write("│   └── ana_stats.json        # Analysis statistics JSON file\n")
            f.write("└── README.md                 # This document\n")
            f.write("```\n\n")
            f.write("## File Descriptions\n\n")
            f.write(
                "- **ana_mean.bin**: Mean of ensemble analysis, for post-processing and evaluation\n"
            )
            f.write("- **ana_ensemble/**: Analysis field for each ensemble member, used for restarts\n")
            f.write(
                "- **ana_stats/**: Statistical diagnostics of the analysis ensemble\n\n"
            )
            f.write("## Restart Notes\n\n")
            f.write(
                "During restart, each ensemble member uses its analysis field (ensemble_XX.bin) as initial condition,\n"
            )
            f.write("which preserves ensemble diversity and avoids degeneration.\n\n")
            f.write("## Statistics\n\n")
            f.write(f"- Ensemble size: {self.Nensemble}\n")
            f.write(f"- Number of detectors: {len(self.detector_list)}\n")
            f.write(f"- Domain size: {self.current_result_size}\n")
            f.write(
                f"- Analysis mean range: [{stats_info['analysis_mean_min']:.6e}, {stats_info['analysis_mean_max']:.6e}]\n"
            )
            f.write(
                f"- Analysis ensemble variance range: [{stats_info['analysis_ensemble_variance_min']:.6e}, {stats_info['analysis_ensemble_variance_max']:.6e}]\n"
            )

    def update_result_size(self):
        self.model.update_current_result_size_after_restart()
        self.current_result_size = self.model.get_current_result_size()

    def restart_run(self):
        
        if not self.verify_restart_folder(self.current_time_step):
            print(
                f"Error: Restart folder {self.current_time_step} failed validation; cannot proceed",
                file=sys.stderr,
            )
            sys.exit(1)

        
        restart_info = self.get_restart_info(self.current_time_step)
        if restart_info:
            print(
                f"Restart info: time step {restart_info['time_step']}, ensemble size {restart_info['ensemble_size']}"
            )
            print(f"Number of detectors: {restart_info['detector_count']}")
            print(
                f"Analysis mean range: [{restart_info['analysis_mean_min']:.6e}, {restart_info['analysis_mean_max']:.6e}]"
            )

        
        self.model.restart_config_update()
        self.model.run_forward()
        self.update_result_size()
        self.model.update_output_keyarealist()
        if self.inverse_iteration:
            self.model.update_keyarealist()
            self.model.run_backward()

        print(f"✓ Restart for time step {self.current_time_step} completed")

    def dump_detection(self):
        self.detection_dir.mkdir(parents=True, exist_ok=True)
        file_name = self.detection_dir / (str(self.current_time_step) + ".json")
        simulation, assimilation = self.get_simulation_and_assimilation()
        with open(file_name, "w", encoding="utf-8") as f:
            detector_dict = {}
            for i, (lon, lat, height) in enumerate(self.tran_geo_detector_list):
                detector_dict[f"detector_{i}"] = {}
                detector_dict[f"detector_{i}"]["lon"] = lon
                detector_dict[f"detector_{i}"]["lat"] = lat
                detector_dict[f"detector_{i}"]["height"] = height
                detector_dict[f"detector_{i}"]["simulation"] = {}
                for j, si in enumerate(simulation[i]):
                    detector_dict[f"detector_{i}"]["simulation"][f"ensemble_{j}"] = (
                        float(si)
                    )
                detector_dict[f"detector_{i}"]["detection"] = float(
                    self.detector_values[i]
                )
                detector_dict[f"detector_{i}"]["assimilation"] = float(assimilation[i])
            json.dump(detector_dict, f, ensure_ascii=False, indent=4)

    def dump_conclist(self):
        list_dir = self.key_message_dir / "conclist"
        list_dir.mkdir(parents=True, exist_ok=True)
        file_name = list_dir / (str(self.current_time_step) + ".json")
        with open(file_name, "w", encoding="utf-8") as f:
            conc_dict = {}
            for keyarea in self.model.get_keyarea_list():
                name = keyarea.get_areaname()
                conc_dict[name] = {}
                conc_list = keyarea.get_conclist_for_output()
                for i, conclist in enumerate(conc_list):
                    conc_dict[name][f"ensemble_{i}"] = conclist.tolist()
            json.dump(conc_dict, f, ensure_ascii=False, indent=4)

    def dump_keyarea_message(self):
        self.key_message_dir.mkdir(parents=True, exist_ok=True)
        file_name = self.key_message_dir / (str(self.current_time_step) + ".json")
        keyarealist = self.model.get_keyarea_list()
        with open(file_name, "w", encoding="utf-8") as f:
            message_dict = {}
            for keyarea in keyarealist:
                name = keyarea.get_areaname()
                message_dict[name] = {}
                meanconc = keyarea.get_meanconc_for_output()
                message_dict[name]["meanconc"] = {}
                for i, c in enumerate(meanconc):
                    message_dict[name]["meanconc"][f"ensemble_{i}"] = c
                message_dict[name]["arrivetime"] = {}
                arrivetime = keyarea.get_arrivetime_for_output()
                for i, t in enumerate(arrivetime):
                    t_str = t.strftime("%Y-%m-%d_%H-%M-%S")
                    message_dict[name]["arrivetime"][f"ensemble_{i}"] = t_str
            json.dump(message_dict, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)

    def dump_p_field(self, P_forward, P_backward, P_field):
        
        if not self.flag_dump_p_field:
            return
        
        dump_path = self.model.get_dump_path()
        p_field_dir = dump_path / "P_field" / str(self.current_time_step)
        p_field_dir.mkdir(parents=True, exist_ok=True)
        
        forward_file = p_field_dir / "forward_P_field.bin"
        write_data(forward_file, P_forward)
        
        p_field_file = p_field_dir / "P_field.bin"
        write_data(p_field_file, P_field)
        
        if P_backward is not None:
            backward_file = p_field_dir / "backward_P_field.bin"
            write_data(backward_file, P_backward)
        
        print(f"[P_field] saved to: {p_field_dir}")

    def read_fixed_stations(self):
        self.tran_geo_detector_list = []
        self.detector_list = []
        with open(self.info_station_dir, "r", encoding="utf-8") as f:
            stationlist = json.load(f)
        for station in stationlist:
            lon = float(station["lon"])
            lat = float(station["lat"])
            heigh = 15  
            self.tran_geo_detector_list.append((lon, lat, heigh))
            cor_list = geo2cordinate(lon, lat, self.domain_size)
            x = round(cor_list[0])
            y = round(cor_list[1])
            z = int(0)
            self.detector_list.append((x, y, z))

    def run_assimilation(self):
        for step in self.assimulation_step_list:
            self.update_step(step)
            self.initial_state()
            if self.flag_fixed_station:
                if self.read_station:
                    self.read_fixed_stations()
                    self.read_station = False
            else:
                self.update_detectors()
            self.assimilation()
            if self.flag_dump_detection:
                self.dump_detection()

            self.dump_ana_result()

            self.restart_run()

            if self.flag_dump_keyarea:
                self.dump_keyarea_message()
                self.dump_conclist()

            self.last_assimlation_step = step

    def print_detectors(self):
        print("The detector list is:")
        for i, detector in enumerate(self.detector_list):
            print(f"Detector {i}: {detector}")

    def get_detectors(self):
        if self.detector_list is None:
            print("The detector list has not been initialized.", file=sys.stderr)
            sys.exit(1)
        return self.detector_list

    def get_domain_size(self):
        if self.domain_size is None:
            print("The domain size has not been initialized.", file=sys.stderr)
            sys.exit(1)
        return self.domain_size

    def verify_restart_folder(self, time_step):
        
        restart_path = self.model.get_dump_path() / str(time_step)

        if not restart_path.exists():
            print(f"Error: Restart folder does not exist: {restart_path}", file=sys.stderr)
            return False

        
        required_files = [
            restart_path / "ana_mean.bin",
            restart_path / "ana_ensemble",
            restart_path / "ana_stats",
            restart_path / "README.md",
        ]

        for file_path in required_files:
            if not file_path.exists():
                print(f"Error: Required file missing: {file_path}", file=sys.stderr)
                return False

        
        ana_ensemble_path = restart_path / "ana_ensemble"
        for i in range(self.Nensemble):
            ensemble_file = ana_ensemble_path / f"ensemble_{i:02d}.bin"
            if not ensemble_file.exists():
                print(
                    f"Error: Analysis field for ensemble member {i} not found: {ensemble_file}",
                    file=sys.stderr,
                )
                return False

        
        stats_path = restart_path / "ana_stats"
        required_stats_files = [
            "ana_variance.bin",
            "ana_std.bin",
            "ana_cv.bin",
            "ana_stats.json",
        ]

        for stats_file in required_stats_files:
            if not (stats_path / stats_file).exists():
                print(f"Error: Statistics file missing: {stats_file}", file=sys.stderr)
                return False

        print(f"✓ Restart folder {time_step} verification passed")
        return True

    def get_restart_info(self, time_step):
        
        restart_path = self.model.get_dump_path() / str(time_step)
        stats_json_path = restart_path / "ana_stats" / "ana_stats.json"

        if not stats_json_path.exists():
            print(f"Error: Statistics JSON file not found: {stats_json_path}", file=sys.stderr)
            return None

        try:
            with open(stats_json_path, "r", encoding="utf-8") as f:
                stats_info = json.load(f)
            return stats_info
        except Exception as e:
            print(f"Error: Failed to read statistics JSON: {e}", file=sys.stderr)
            return None
