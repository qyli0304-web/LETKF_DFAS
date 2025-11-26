"""
Utilities for generating geographic masks from country boundaries.
Provides a fast vectorized mask generator and a fallback point-in-polygon
implementation using GeoPandas and Shapely.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import geodatasets
from shapely.vectorized import contains

def generate_country_mask_fast(lon_min, lat_min, Nx, Ny, dx, dy, country_name="China"):
    
    lons = lon_min + np.arange(Nx) * dx
    lats = lat_min + np.arange(Ny) * dy
    lons_grid, lats_grid = np.meshgrid(lons, lats)  # (Ny, Nx)

    country = load_country_shape(country_name)
    polygon = country.geometry.unary_union  


    mask = contains(polygon, lons_grid, lats_grid).astype(int)
    return mask, lons_grid, lats_grid 
def generate_country_mask(lon_min, lat_min, Nx, Ny, dx, dy, country_name="China"):
   
    
    lons = lon_min + np.arange(Nx) * dx
    lats = lat_min + np.arange(Ny) * dy
    lons_grid, lats_grid = np.meshgrid(lons, lats)  # (Ny, Nx)

   
    country = load_country_shape(country_name)  

   
    mask = np.zeros((Ny, Nx), dtype=int) 
    for i in range(Ny):
        for j in range(Nx):
            point = Point(lons_grid[i, j], lats_grid[i, j])
            if country.geometry.contains(point).any():
                mask[i, j] = 1

    return mask, lons_grid, lats_grid


def load_country_shape(country_name):
    
    world = gpd.read_file("/home/liqy/ETKF/ne_geo/ne_110m_admin_0_countries.shp")
    
    possible_columns = ["NAME", "name", "ADMIN", "COUNTRY"]
    for col in possible_columns:
        if col in world.columns:
            country = world[world[col] == country_name]
            if len(country) > 0:
                return country
    
    raise ValueError(f"Can not find '{country_name}' country in the shapefile: {world.columns}")


if __name__ == "__main__":
    
    lon_min = 70.0   
    lat_min = 15.0   
    Nx = 100         
    Ny = 80          
    dx = 0.5         
    dy = 0.5         

    mask, lons, lats = generate_country_mask_fast(lon_min, lat_min, Nx, Ny, dx, dy, "China")

    print("Mask shape:", mask.shape) 
    print("Mask sample:\n", mask[:5, :5]) 

    import matplotlib.pyplot as plt

    plt.imshow(mask, extent=(lon_min, lon_min + Nx*dx, lat_min, lat_min + Ny*dy), origin="lower")
    plt.colorbar(label="Mask (1=China)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("China Territory Mask")
    plt.show()


