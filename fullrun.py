"""
Orchestrates a complete data assimilation run using the configured model,
observation manager, and assimilation driver. The script initializes all
components from configuration files and executes the assimilation workflow.
"""
from src.model import Model
from src.obs_manager import ObsManager
from src.Assimulation import AssimilationDriver
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from src.utils import *
model = Model(config_file="model.cfg")
obs = ObsManager(config_file="obs.cfg")
assimilation = AssimilationDriver(config_file="assimulation.cfg")
assimilation.initialize(model, obs)
assimilation.run_assimilation()
