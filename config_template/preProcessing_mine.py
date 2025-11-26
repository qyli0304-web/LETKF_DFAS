"""
Preprocessing driver for Polyphemus/Polair3D inputs. Generates ground, roughness,
meteorology (including Kz and Kz_TM), and deposition inputs over a date range,
then executes the model. Paths are environment-specific and should be updated.
"""
import os
from pathlib import Path
from datetime import date, timedelta

startDay = date(2023, 10, 14)
totalDays = 12
current_file = Path(__file__).resolve()
current_dir = current_file.parent
groundFile = Path("./data/ground/LUC-glcf.bin")

if not groundFile.is_file():
    groundPath = Path("./data/ground/")
    groundPath.mkdir(parents=True, exist_ok=True)
    os.system("/home/liqy/workAPP/Polyphemus-1.11.1/preprocessing/ground/luc-glcf config/general.cfg config/luc-glcf.cfg")

groundZhangFile = Path("./data/ground/LUC-glcf-zhang.bin")
if not groundZhangFile.is_file():
    os.system("/home/liqy/workAPP/Polyphemus-1.11.1/preprocessing/ground/luc-convert  config/general.cfg config/glcf_to_zhang.cfg")

roughnessFile = Path("./data/ground/Roughness-glcf.bin")
if not roughnessFile.is_file():
    os.system("/home/liqy/workAPP/Polyphemus-1.11.1/preprocessing/ground/roughness config/general.cfg config/roughness.cfg")

meteoPath = Path("./data/meteo/")
meteoPath.mkdir(parents=True, exist_ok=True)

kzTMPath = Path("./data/meteo/Kz_TM")
kzTMPath.mkdir(parents=True, exist_ok=True)

depPath = Path("./data/dep")
depPath.mkdir(parents=True, exist_ok=True)

resultsPath = Path("./results")
resultsPath.mkdir(parents=True, exist_ok=True)

for daysN in range(0,totalDays):
    durations = timedelta(days=daysN)
    currentDay = durations+startDay
    nextDay = currentDay+timedelta(days=1)
    print('Current Day:' + currentDay.isoformat())

    os.system("/home/liqy/workAPP/Polyphemus-1.11.1/preprocessing/meteo/meteo_parallel config/general.cfg config/meteo.cfg "+currentDay.strftime('%Y-%m-%d'))

    os.system("/home/liqy/workAPP/Polyphemus-1.11.1/preprocessing/meteo/Kz config/general.cfg config/meteo.cfg "+currentDay.strftime('%Y-%m-%d'))

    os.system("/home/liqy/workAPP/Polyphemus-1.11.1/preprocessing/meteo/Kz_TM config/general.cfg config/meteo.cfg "+currentDay.strftime('%Y-%m-%d')+" "+nextDay.strftime('%Y-%m-%d'))

    os.system("/home/liqy/workAPP/Polyphemus-1.11.1/preprocessing/dep/dep_aerosol config/general.cfg config/dep.cfg "+currentDay.strftime('%Y-%m-%d')+" "+nextDay.strftime('%Y-%m-%d'))

os.system("/home/liqy/workAPP/Polyphemus-1.11.1/processing/photochemistry/polair3d config/case.cfg")
