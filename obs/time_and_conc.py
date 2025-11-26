"""
Derive arrival times and concentration statistics for key locations based on
simulated concentration fields. The script reads gridded data, samples at
specified city locations and heights, and writes a JSON report containing
time series, mean concentrations, and first-arrival timestamps.
"""
from src.polyphemus_config import ConfigParser
from src.utils import *
from datetime import datetime,timedelta
import json

def get_conc_list(x, y, z, conc):
    """Return the concentration time series at grid index (x, y, z)."""
    return conc[x,y,z,:]

def mean_list(list):
    """Compute the mean of non-negative values in an array-like sequence."""
    nonzero_list = list[list>=0]
    if len(nonzero_list)>0:
        return nonzero_list.mean()
    else:
        return 0

def cal_arrive_index(lst, value):
    """Return the first index where the series exceeds a threshold value."""
    arrive_flag = 0
    for index in range(len(lst)):
        if arrive_flag == 0:
            if lst[index] >= value:
                    arrive_index = index
                    arrive_flag = 1
            else:
                continue
    return arrive_index

start_time = datetime(2023,10,15,0,0)
limited_value = 0.001
domain_size = [70.125,15.125,0.25,0.25]
file_path = 'obs/obs.bin'
output_path = "obs/arrive.json"


levels = np.loadtxt('config_template/levels.dat').tolist()[1:]

city_config = ConfigParser()
city_config.parse('keyarea.cfg')
city_list = city_config.get_section()
conc = read_data(file_path,[280,180,10,240])

mydict = {}
for value in city_list:
    name = value['Name']
    mydict[name] = {}
    lon = float(value['Abscissa'])
    lat = float(value['Ordinate'])
    height = float(value['Altitude'])
    x,y = geo2cordinate(lon,lat,domain_size)
    z = np.searchsorted(levels, height, side='right')
    x = round(x)
    y = round(y)
    z = int(z)
    conc_lst = get_conc_list(x,y,z,conc)
    mydict[name]['conc_list'] = conc_lst.tolist()
    meanconc = mean_list(conc_lst)
    mydict[name]['mean_conc'] = str(meanconc)
    index = cal_arrive_index(conc_lst,limited_value)
    arrivetime = start_time+(index+1)*timedelta(hours=1)
    arrivetime_str = arrivetime.strftime('%Y-%m-%d_%H-%M-%S')
    mydict[name]['arrive_time'] = arrivetime_str

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(mydict, f, indent=2)
