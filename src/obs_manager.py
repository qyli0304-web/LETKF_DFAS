"""
Observation manager for sampling gridded concentration data at arbitrary
geospatial points and time steps. Handles configuration, vertical level
mapping, and interpolation onto detector locations.
"""
import numpy as np
from src.polyphemus_config import ConfigParser
from src.utils import *
from pathlib import Path
from scipy.ndimage import map_coordinates
import sys

class ObsManager:
    """Manage observation configuration and sampling of gridded fields."""
    def __init__(self, config_file):
        self.config_file = ConfigParser()
        self.config_file.parse(config_file)
        self.obs_path = Path(self.config_file.get('path')['obs_file_path'])
        self.R = float(self.config_file.get('detector')['R'])
        self.obsvalue = None
        self.trueconcsize = [int(self.config_file.get('domain')['Nx']),
                             int(self.config_file.get('domain')['Ny']),
                             int(self.config_file.get('domain')['Nz']),
                             int(self.config_file.get('domain')['Nt'])]
        self.domain_parameters = [
            float(self.config_file.get('domain')['x_min']),
            float(self.config_file.get('domain')['y_min']),
            float(self.config_file.get('domain')['Delta_x']),
            float(self.config_file.get('domain')['Delta_y'])
        ]
        level_path = self.config_file.get('path')['level_path']
        self.levels = np.loadtxt(level_path).tolist()[1:]
        
    def get_obs_var(self):
        """Return the scalar observation error variance R."""
        if self.R is None:
            print('The observation var has not been initialized.', file=sys.stderr)
            sys.exit(1)
        else:
            return self.R

    def calculate_relative_coordinates(self, points):
        """Convert lon-lat-height points to relative grid coordinates.

        Parameters
        ----------
        points : array-like of shape (m, 3) or (3,)
            Geographic points [lon, lat, height].

        Returns
        -------
        np.ndarray
            Relative coordinates in grid index space (x, y, z_index).
        """
        points = np.atleast_2d(points)
        lon_min, lat_min, dx, dy = self.domain_parameters
        relative_lon = (points[:, 0] - lon_min) / dx
        relative_lat = (points[:, 1] - lat_min) / dy
        heights = points[:, 2]
        relative_height = np.searchsorted(self.levels, heights, side='right')
        relative_coords = np.column_stack((relative_lon, relative_lat, relative_height))
        if relative_coords.shape[0] == 1:
            return relative_coords[0]
        else:
            return relative_coords

    def cal_obsvalue(self, obs_location, step):
        """Sample gridded concentration at provided locations for a given step.

        Returns the sampled values stored internally via `get_obsvalue`.
        """
        true_value = read_data_steps(self.obs_path, self.trueconcsize, step, 1).squeeze()
        relative_location = self.calculate_relative_coordinates(obs_location)
        rel_coords = np.atleast_2d(relative_location)
        z_layers = np.clip(np.round(rel_coords[:, 2]).astype(int), 0, self.trueconcsize[2]-1)
        xi = np.clip(rel_coords[:, 0], 0, self.trueconcsize[0]-1)
        yi = np.clip(rel_coords[:, 1], 0, self.trueconcsize[1]-1)
        coords = np.vstack([xi, yi])
        detects =  []
        for z in np.unique(z_layers):
            mask = z_layers == z
            if mask.any():
                vals = map_coordinates(true_value[:, :, z], 
                                 coords[:, mask], 
                                 order=1)
                detects.append((mask, vals))
        out = np.zeros(len(rel_coords))
        for mask, vals in detects:
            out[mask] = vals
        if len(out) > 1:
            self.obsvalue = out
        else:
            self.obsvalue = out[0]

    def get_obsvalue(self):
        """Return the last computed observation values."""
        return self.obsvalue
