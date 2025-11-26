"""
Utility functions for reading/writing Fortran-order binary grids, coordinate
transformations, ensemble statistics, and plotting helpers.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import os, json, math, imageio, rasterio
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from matplotlib.ticker import LogFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
import shutil

def normalize_data(data):
    """Normalize an array to the range [0, 1]."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        return np.zeros_like(data)  
    return (data - data_min) / (data_max - data_min)

def copy_with_overwrite(src, dst):
    """Recursively copy a directory tree from src to dst, overwriting files."""
    if not os.path.exists(dst):
        os.makedirs(dst)  

    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.isdir(src_path):
            
            copy_with_overwrite(src_path, dst_path)
        else:
           
            shutil.copy2(src_path, dst_path)

def load_inverse_ensemble(
    pathname: str,
    *,
    n_ensemble: int,
    n_step: int,
    sizes: list,
    city_list: list,
    filename: str
) -> dict:
    """Load inverse ensemble results for multiple cities into a dict."""
    ensemble_list = {}
    
    for i, city in enumerate(city_list):
        data_list = []
        for j in range(n_ensemble):
            air_conc_path = Path(pathname)/str(j)/'results'/(city+filename)
            data_list.append(read_data_steps(air_conc_path, sizes, n_step, 1))
        ensemble_list[city] = np.squeeze(np.array(data_list),axis= -1)
    return ensemble_list

def load_ensemble(
    pathname: str,
    *,
    n_ensemble: int,
    n_step: int,
    sizes: list,
    filename: str = 'ensemble_Cs137_air.bin'
) -> np.array:
    """Load ensemble results from per-member result folders."""
    ensemble_list = []
    for i in range(n_ensemble):
        air_conc_path = Path(pathname)/str(i)/'results'/filename
        ensemble_list.append(read_data_steps(air_conc_path, sizes, n_step, 1))
    return np.squeeze(np.array(ensemble_list), axis=-1)
def calculate_cv_field(array):
    """Compute coefficient-of-variation field across ensembles, safe at zero mean."""
    
    mean_field = np.mean(array, axis=0)
    
    std_field = np.std(array, axis=0, ddof=1) 
    
    
    cv_field = np.zeros_like(mean_field)
    
    non_zero_mask = np.abs(mean_field) > 1e-10 
    cv_field[non_zero_mask] = std_field[non_zero_mask] / np.abs(mean_field[non_zero_mask])
    
    return cv_field
def cal_variance(
        ensemble_array: np.array
):
    """Sample variance along the ensemble dimension (ddof=1)."""
    variance = np.var(ensemble_array, axis=0, ddof=1)
    return variance

def cordinate2geo(x, y, domain_size):
    """Map grid coordinate (x, y) to geographic lon/lat using domain parameters."""
    lon = x * domain_size[2] + domain_size[0]
    lat = y * domain_size[3] + domain_size[1]
    return lon, lat

def geo2cordinate(lon,lat,domain_size):
    """Map geographic lon/lat to fractional grid coordinate [x, y]."""
    ord_index=[]
    x=(lon-domain_size[0])/domain_size[2]
    ord_index.append(x)
    y=(lat-domain_size[1])/domain_size[3]
    ord_index.append(y)
    return ord_index

def write_data(filename, data):
    """Write array in Fortran-order as float32 binary to filename."""
    data_f = data.flatten(order='F').astype(np.float32)  
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        data_f.tofile(f)

def read_data(filename, sizes):
    """Read float32 Fortran-order binary into (Nx, Ny, Nz, Nt)."""
    Nx, Ny,Nz, Nt = sizes[0], sizes[1], sizes[2], sizes[3]
    data_n = Nx * Ny * Nz * Nt
    
    with open(filename, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32, count=data_n)
    
    return data.reshape((Nx, Ny, Nz, Nt), order='F')

def read_data_interval(filename, sizes, start_time, end_time):
    """Read a time interval [start_time, end_time) from a 4D field file."""
    Nx, Ny, Nz, Nt = sizes[0], sizes[1], sizes[2],sizes[3]
    data_n = Nx * Ny * Nz * (end_time - start_time)
    with open(filename, 'rb') as file:
        
        file.seek((start_time * Nx * Ny * Nz) * 4)
        data = np.fromfile(file, dtype=np.float32, count=data_n)
    
    return data.reshape((Nx, Ny, Nz, end_time - start_time), order='F')

def read_data_steps(filename, sizes, start, blocknumber):
    """Read a number of time steps starting at index start from a 4D file."""
    Nx, Ny, Nz, Nt = sizes[0], sizes[1], sizes[2],sizes[3]
    data_n = Nx * Ny * Nz * blocknumber
    with open(filename, 'rb') as file:
       
        file.seek((start * Nx * Ny * Nz) * 4)
        data = np.fromfile(file, dtype=np.float32, count=data_n)
    
    return data.reshape((Nx, Ny, Nz, blocknumber), order='F')



def read_conc_data(filename, sizes):
    """Read a 3D (Nx, Ny, Nt) concentration file in Fortran order."""
    Nx, Ny, Nt = sizes[0], sizes[1], sizes[3]
    data_n = Nx * Ny * Nt
    
    with open(filename, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32, count=data_n)
    
    return data.reshape((Nx, Ny, Nt), order='F')

def read_conc_data_step(filename, sizes, start_time, end_time):
    """Read a time window from a 3D (Nx, Ny, Nt) concentration file."""
    Nx, Ny, Nt = sizes[0], sizes[1], sizes[3]
    
    data_n = Nx * Ny * (end_time - start_time)
    
    with open(filename, 'rb') as file:
        
        file.seek((start_time * Nx * Ny) * 4)  
        
        data = np.fromfile(file, dtype=np.float32, count=data_n)
    
    
    return data.reshape((Nx, Ny, end_time - start_time), order='F')

def read_conc_onepoint(filename, sizes,start,blocknumber):
    """Read a subset of time steps from a 3D (Nx, Ny, Nt) file."""
    Nx, Ny, Nt = sizes[0], sizes[1], sizes[3]
    data_n = Nx * Ny * blocknumber
    with open(filename, 'rb') as file:
        
        file.seek((start * Nx * Ny) * 4)  
        
        data = np.fromfile(file, dtype=np.float32, count=data_n)
    
    
    return data.reshape((Nx, Ny, blocknumber), order='F')

def compute_group_covariance(data):
    """Compute covariance matrices across groups for each time slice."""
    G, Nx, Ny, Nt = data.shape
    cov_matrices = []

    for t in range(Nt):  
        spatial_data = data[:, :, :, t].reshape(G, Nx * Ny)
        
        
        
        cov_matrices.append(cov_matrix)

    return np.array(cov_matrices)  

if __name__ =='__main__':
    sizeofbin = [140, 90, 1, 768]
    name=Path('Kz_Louis.bin')
    name_copy=Path('Kz_Louis_copy.bin')
    R=read_data(name,sizeofbin)
    write_data(name_copy,R)
    loaded_data = read_data(name_copy, sizeofbin)
    assert np.array_equal(R, loaded_data)
    print("Read/write check passed.")
