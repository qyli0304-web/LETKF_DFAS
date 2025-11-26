"""
Key area entity for aggregating time series and arrival times at specified
locations. Provides utilities to compute per-ensemble concentration lists,
mean concentrations, and first arrival times given thresholds.
"""
from src.utils import *
import math
class keyarea:
    def __init__(self,area_dict):
        self.name = area_dict['Name']
        self.population = float(area_dict['Population'])
        self.abscissa = float(area_dict['Abscissa'])
        self.ordinate = float(area_dict['Ordinate'])
        self.conclist = None
        self.arrivetime = None
        self.meanconc = None
        self.cordination = None
        self.altitude =float(area_dict['Altitude'])

        self.conclist_for_output = None
        self.meanconc_for_output = None
        self.arrivetime_for_output = None
    
    def cal_conclist(self,Nensemble, result_path, domain_size,size,filename):
        """Compute per-ensemble concentration time series at the area location."""
        self.conclist = []
        for i in range(Nensemble):
            
            self.cordination = geo2cordinate(self.abscissa,self.ordinate,domain_size)
            target_path = result_path / str(i) / 'results' / filename
            self.conclist.append(read_data(target_path,size)[math.floor(self.cordination[0]),math.floor(self.cordination[1]),0,:])
    
    def get_conclist(self):
        return self.conclist
    

    def cal_conclist_for_output(self,Nensemble, result_path, domain_size,size,filename):
        """Compute per-ensemble time series for output reporting."""
        self.conclist_for_output = []
        for i in range(Nensemble):
            
            self.cordination = geo2cordinate(self.abscissa,self.ordinate,domain_size)
            target_path = result_path / str(i) / 'results' / filename
            self.conclist_for_output.append(read_data(target_path,size)[math.floor(self.cordination[0]),math.floor(self.cordination[1]),0,:])
    
    def get_conclist_for_output(self):
        return self.conclist_for_output

    def cal_meanconc(self):
        """Compute mean concentration across non-zero entries for each ensemble."""
        self.meanconc = []
        for list in self.conclist:
            non_zero_data = list[list != 0]
            if len(non_zero_data) > 0:  # 检查非空
                non_zero_mean = non_zero_data.mean()
            else:
                non_zero_mean = 0.0 
            self.meanconc.append(non_zero_mean)
    
    def get_meanconc(self):
        return self.meanconc
    
    def cal_meanconc_for_output(self):
        """Compute mean concentration for output time series per ensemble."""
        self.meanconc_for_output = []
        for list in self.conclist_for_output:
            non_zero_data = list[list != 0]
            if len(non_zero_data) > 0:  # 检查非空
                non_zero_mean = non_zero_data.mean()
            else:
                non_zero_mean = 0.0 
            self.meanconc_for_output.append(non_zero_mean)
    
    def get_meanconc_for_output(self):
        return self.meanconc_for_output
    
    def cal_arrivetime(self, limitvalue, simulation_begin, sim_save_deltat,step):
        """Compute first-arrival times crossing threshold for each ensemble."""
        self.arrivetime = []
        for item in self.conclist:
            arrive_flag = 0
            
            for index in range(len(item)):
                if arrive_flag ==0:
                    if item[index] >= limitvalue:
                        arrive_index = index
                        arrive_flag = 1
                else:
                    continue
            if arrive_flag ==1:
                self.arrivetime.append(simulation_begin+(arrive_index+step+1)*sim_save_deltat*timedelta(hours=1))
            else:
                self.arrivetime.append(simulation_begin+(len(item)+step+1)*sim_save_deltat*timedelta(hours=1))
    
    def get_arrivetime(self):
        return self.arrivetime
    
    def cal_arrivetime_for_output(self, limitvalue, simulation_begin, sim_save_deltat,step):
        """Compute first-arrival times for output series per ensemble."""
        self.arrivetime_for_output = []
        for item in self.conclist_for_output:
            arrive_flag = 0
            
            for index in range(len(item)):
                if arrive_flag ==0:
                    if item[index] >= limitvalue:
                        arrive_index = index
                        arrive_flag = 1
                else:
                    continue
            if arrive_flag ==1:
                
                self.arrivetime_for_output.append(simulation_begin+(arrive_index+step+1)*sim_save_deltat*timedelta(hours=1))
            else:
                self.arrivetime_for_output.append(simulation_begin+(len(item)+step+1)*sim_save_deltat*timedelta(hours=1))
    
    def get_arrivetime_for_output(self):
        return self.arrivetime_for_output
    
    def get_areaname(self):
        return self.name
    
    def get_abscissa(self):
        return self.abscissa
    
    def get_ordinate(self):
        return self.ordinate
    
    def get_altitude(self):
        return self.altitude
    
    def get_population(self):
        return self.population
            
