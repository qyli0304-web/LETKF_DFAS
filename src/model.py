"""
Model orchestration for forward and backward Polyphemus/Polair3D runs,
configuration management, ensemble directory setup, and key-area utilities.
"""
import numpy as np
from src.polyphemus_config import ConfigParser
from datetime import datetime, timedelta
import os
import subprocess
from pathlib import Path
from src.utils import *
import copy
import math
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


class Model:
    """Manage simulation configuration, run orchestration, and key-area data."""
    def __init__(self, config_file):
        
        self.current_time_step = 0

        self.runconfig = ConfigParser()
        self.runconfig.parse(config_file)
        self.runconfig.resolve_markups()
        self.forward_name = self.runconfig.get('name')['forward_name']
        self.release_name = self.runconfig.get('name')['release_name']
        self.forward_conc_filename = self.forward_name+'_'+self.release_name+'_air.bin'
        
        
        self.main_config = ConfigParser()
        self.main_config.parse(self.runconfig.get('path')['config_template_path']+'/case.cfg')
        

        self.data_config = ConfigParser()
        self.data_config.parse(self.runconfig.get('path')['config_template_path']+'/case-data.cfg')


        self.saver_config = ConfigParser()
        self.saver_config.parse(self.runconfig.get('path')['config_template_path']+'/case-saver.cfg')


        self.general_config = ConfigParser()
        self.general_config.parse(self.runconfig.get('path')['config_template_path']+'/general.cfg')

        self.meteo_config = ConfigParser()
        self.meteo_config.parse(self.runconfig.get('path')['config_template_path']+'/meteo.cfg')

        self.source_config = ConfigParser()
        self.source_config.parse(self.runconfig.get('path')['config_template_path']+'/source.dat')

        self.ground_config = ConfigParser()
        self.ground_config.parse(self.runconfig.get('path')['config_template_path']+'/luc-glcf.cfg')
        self.backward_config_files = ['case.cfg','case-data.cfg','case-saver.cfg','levels.dat','source.dat','species.dat']

        
        self.origin_result_size = [int(self.runconfig.get('simulation_domain')['Nx']),\
                            int(self.runconfig.get('simulation_domain')['Ny']),\
                            int(self.runconfig.get('simulation_domain')['Nz']),\
                            int(int(self.runconfig.get('simulation_domain')['Nt'])/int(self.runconfig.get('simulation_domain')['Interval_length']))]

        self.domain_size = [float(self.runconfig.get('simulation_domain')['x_min']),\
                            float(self.runconfig.get('simulation_domain')['y_min']),\
                            float(self.runconfig.get('simulation_domain')['Delta_x']),\
                            float(self.runconfig.get('simulation_domain')['Delta_y'])]
        self.current_result_size = self.origin_result_size
        self.species = self.runconfig.get('release')['specie']
        self.effective_value_flag = (self.runconfig.get('flags')['effective_conc_value'].lower() == 'true')
        self.limitvalue =None
        if self.effective_value_flag:
            self.limitvalue = float(self.runconfig.get('limit')['effective_conc_value'])
        self.simulation_date_min = None
        self.simulation_date_max = None
        self.sim_save_delta_t = float(self.runconfig.get('simulation_domain')['Delta_t'])*int(self.runconfig.get('simulation_domain')['Interval_length'])/3600.0 ###hours
        self.simulation_delta_t = float(self.runconfig.get('simulation_domain')['Delta_t'])
        self.simulation_nt = int(self.runconfig.get('simulation_domain')['Nt'])
        self.keyarealist = None
        self.Redundant_day = None
        self.sequential_release_file = None
        self.sequential_release_path = None
        self.release_scalor_factor = None
        self.release_change_flag = None
        self.random_seed = None
        self.backward_model = int(self.runconfig.get('backward')['inverse_mode'])
        

        self.Nensemble = int(self.runconfig.get('simulation_domain')['Nensemble'])
        self.polyphemus_path = os.path.abspath(self.runconfig.get('path')['polyphemus_path'])
        self.forward_dir = Path(self.runconfig.get('path')['forward_folder_path']).resolve()
        self.backward_dir = Path(self.runconfig.get('path')['backward_folder_path']).resolve()
        self.restart_dir = Path(self.runconfig.get('path')['restart_path']).absolute()

       

    def initialize(self):
        """Initialize configuration files, synchronize run directories, and prepare meteo data."""
        simulation_date_min_str = self.runconfig.get('simulation_domain')['Date_min']##2023-10-15_00-00-00
        self.simulation_date_min = datetime.strptime(simulation_date_min_str, '%Y-%m-%d_%H-%M-%S')
        self.simulation_date_max = self.simulation_date_min + timedelta(seconds=self.simulation_nt * self.simulation_delta_t)
        self.main_config.set('domain','Date_min', simulation_date_min_str)
        self.main_config.set('domain','Delta_t', self.runconfig.get('simulation_domain')['Delta_t'])
        self.main_config.set('domain','Nt', self.runconfig.get('simulation_domain')['Nt'])
        self.main_config.set('domain','x_min', self.runconfig.get('simulation_domain')['x_min'])
        self.main_config.set('domain','Delta_x', self.runconfig.get('simulation_domain')['Delta_x'])
        self.main_config.set('domain','Nx', self.runconfig.get('simulation_domain')['Nx'])
        self.main_config.set('domain','y_min', self.runconfig.get('simulation_domain')['y_min'])
        self.main_config.set('domain','Delta_y', self.runconfig.get('simulation_domain')['Delta_y'])
        self.main_config.set('domain','Ny', self.runconfig.get('simulation_domain')['Ny'])
        self.main_config.set('domain','Nz', self.runconfig.get('simulation_domain')['Nz'])
        self.main_config.set('options','With_initial_condition','no')

        self.Redundant_day = int(self.runconfig.get('meteo')['Redundant_meteo'])
        self.slice_h = int(int(self.runconfig.get('simulation_domain')['Interval_length']) * float(self.runconfig.get('simulation_domain')['Delta_t']) / 3600)
        meteo_date_min = self.simulation_date_min - timedelta(days=self.Redundant_day)
        meteo_date_min_str = meteo_date_min.strftime('%Y-%m-%d_%H-%M-%S')
        self.data_config.set('meteo','Date_min', meteo_date_min_str)
        self.data_config.set('meteo','Delta_t', str(int(float(self.runconfig.get('meteo')['Delta_t_simulation'])*3600.0)))
        self.data_config.set('deposition','Date_min', meteo_date_min_str)
        self.data_config.set('deposition','Delta_t', str(int(float(self.runconfig.get('meteo')['Delta_t_simulation'])*3600.0)))

        self.saver_config.set('save','model',self.forward_name,1)
        self.saver_config.set('save','Interval_length',self.runconfig.get('simulation_domain')['Interval_length'],1)
        self.saver_config.set('save','Interval_length',self.runconfig.get('simulation_domain')['Interval_length'],2)
        self.saver_config.set('save','Interval_length',self.runconfig.get('simulation_domain')['Interval_length'],3)

        self.general_config.set('domain','Date',meteo_date_min_str)
        self.general_config.set('domain','Delta_t',self.runconfig.get('meteo')['Delta_t_simulation'])
        self.general_config.set('domain','x_min', self.runconfig.get('meteo')['x_min_simulation'])
        self.general_config.set('domain','Delta_x', self.runconfig.get('meteo')['Delta_x_simulation'])
        self.general_config.set('domain','Nx', self.runconfig.get('meteo')['Nx_simulation'])
        self.general_config.set('domain','y_min', self.runconfig.get('meteo')['y_min_simulation'])
        self.general_config.set('domain','Delta_y', self.runconfig.get('meteo')['Delta_y_simulation'])
        self.general_config.set('domain','Ny', self.runconfig.get('meteo')['Ny_simulation'])
        self.general_config.set('domain','Nz', self.runconfig.get('meteo')['Nz_simulation'])

        self.meteo_config.set('paths','data_path',self.runconfig.get('path')['ensemble_data_path'])
        self.meteo_config.set('ECMWF','t_min',self.runconfig.get('meteo')['t_min_original'])
        self.meteo_config.set('ECMWF','Delta_t',self.runconfig.get('meteo')['Delta_t_original'])
        self.meteo_config.set('ECMWF','Nt',self.runconfig.get('meteo')['Nt_original'])
        self.meteo_config.set('ECMWF','x_min',self.runconfig.get('meteo')['x_min_original'])
        self.meteo_config.set('ECMWF','Delta_x',self.runconfig.get('meteo')['Delta_x_original'])
        self.meteo_config.set('ECMWF','Nx',self.runconfig.get('meteo')['Nx_original'])
        self.meteo_config.set('ECMWF','y_min',self.runconfig.get('meteo')['y_min_original'])
        self.meteo_config.set('ECMWF','Delta_y',self.runconfig.get('meteo')['Delta_y_original'])
        self.meteo_config.set('ECMWF','Ny',self.runconfig.get('meteo')['Ny_original'])
        self.meteo_config.set('ECMWF','Nz',self.runconfig.get('meteo')['Nz_original'])

        
        self.sequential_release_path = self.runconfig.get('release')['TemporalFactor']
        self.sequential_release_file = os.path.basename(self.sequential_release_path)
        self.release_change_flag = int(self.runconfig.get('release')['flag_random_change_release'])
        self.release_scalor_factor = float(self.runconfig.get('release')['scalor_factor'])
        self.random_seed = int(self.runconfig.get('release')['random_seed'])
        self.source_config.set('source','Abscissa', self.runconfig.get('release')['Abscissa'])
        self.source_config.set('source','Ordinate', self.runconfig.get('release')['Ordinate'])
        self.source_config.set('source','Altitude', self.runconfig.get('release')['Altitude'])
        self.source_config.set('source','Date_beg', self.runconfig.get('release')['Date_beg'])
        self.source_config.set('source','Date_end', self.runconfig.get('release')['Date_end'])
        self.source_config.set('source','TemporalFactor', self.runconfig.get('release')['TemporalFactor'])
        self.source_config.set('source','Date_min_file', self.runconfig.get('release')['Date_min_file'])
        self.source_config.set('source','Delta_t', self.runconfig.get('release')['Delta_t'])

        self.ground_config.set('paths','Database_luc-glcf', self.runconfig.get('path')['ground_data_path'])

        self.main_config.save(self.runconfig.get('path')['config_template_path']+'/case.cfg')
        self.data_config.save(self.runconfig.get('path')['config_template_path']+'/case-data.cfg')
        self.saver_config.save(self.runconfig.get('path')['config_template_path']+'/case-saver.cfg')
        self.general_config.save(self.runconfig.get('path')['config_template_path']+'/general.cfg')
        self.meteo_config.save(self.runconfig.get('path')['config_template_path']+'/meteo.cfg')
        self.source_config.save(self.runconfig.get('path')['config_template_path']+'/source.dat')
        self.ground_config.save(self.runconfig.get('path')['config_template_path']+'/luc-glcf.cfg')

        
    
        
        for i in range(self.Nensemble):
            target_forward_path = self.forward_dir / str(i)
            target_forward_path.mkdir(parents=True, exist_ok=True)
            target_config_path = target_forward_path/'config'
            if not os.path.exists(target_config_path):
                subprocess.run(['cp','-r', self.runconfig.get('path')['config_template_path'], target_config_path], check=True)
            else:
                subprocess.run(['rsync', '-a', '--delete', self.runconfig.get('path')['config_template_path']+'/', target_config_path], check=True)
            meteocfg_path = target_config_path / 'meteo.cfg'
            meteo_config = ConfigParser()
            meteo_config.parse(meteocfg_path)
            meteo_config.set('paths','ensemblenumber', str(i))
            meteo_config.save(meteocfg_path)
        

        if self.release_change_flag != 0 or self.release_scalor_factor !=1.0:
            release_file_path = self.runconfig.get('path')['config_template_path']+'/'+self.sequential_release_file
            filesize = os.path.getsize(release_file_path)  
            count = filesize // 4 
            mutiply_matric = np.ones((self.Nensemble, count))
            if self.release_change_flag == 1:
                np.random.seed(self.random_seed)  
                mutiply_matric = np.random.uniform(0, 2, size=(self.Nensemble, count))
            if self.release_change_flag == 2:
                np.random.seed(self.random_seed) 
                base_random = np.random.rand(self.Nensemble, count) * 2
                w1 = 0.8
                w2 = 0.2
                mutiply_matric = np.zeros((self.Nensemble, count))
                mutiply_matric[:, 0] = base_random[:, 0]
                for j in range(1, count):
                    mutiply_matric[:, j] = w1 * mutiply_matric[:, j-1] + w2 * base_random[:, j]

            mutiply_matric=mutiply_matric*self.release_scalor_factor
            for i in range(self.Nensemble):
                mutiply_array = mutiply_matric[i]
                this_release_file = self.forward_dir / str(i) /'config'/self.sequential_release_file
                release_list = np.fromfile(this_release_file, dtype=np.float32)
                after_muti = release_list*mutiply_array
                after_muti.astype(np.float32).tofile(this_release_file)
        
        for i in range(self.Nensemble):
            target_backward_path = self.backward_dir / str(i)
            target_backward_path.mkdir(parents=True, exist_ok=True)
            backward_config_path = target_backward_path/'config'
            
            if not os.path.exists(backward_config_path):
                backward_config_path.mkdir(parents=True, exist_ok=True)
                for file in self.backward_config_files:
                    subprocess.run(['cp', self.runconfig.get('path')['config_template_path']+'/'+file, backward_config_path], check=True)
            else:
                for file in self.backward_config_files:
                    subprocess.run(['rsync', '-a', '--delete', self.runconfig.get('path')['config_template_path']+'/'+file, backward_config_path], check=True)




        simulation_Nt = int(self.runconfig.get('simulation_domain')['Nt'])
        simulation_delta_t = float(self.runconfig.get('simulation_domain')['Delta_t'])
        self.total_meteo_days = self.Redundant_day*2+int(simulation_Nt*simulation_delta_t/24.0/3600.0)
        forward_meteo_flag = (self.runconfig.get('flags')['ensemble_forward_data'].lower() == 'true')
        if forward_meteo_flag:
            hybrid_coefficients_path = Path(self.runconfig.get('path')['hybrid_meteo_path'])
            meteo_ensemble_path = self.runconfig.get('path')['ensemble_data_path']
            for i in range(self.Nensemble):
                target_meteo_path = os.path.join(meteo_ensemble_path, str(i))
                subprocess.run(["cp", hybrid_coefficients_path, target_meteo_path], check=True)
            meteo_startday = meteo_date_min
            for i in range(self.Nensemble):
                forward_run_path = self.forward_dir / str(i)
                run_script_path_abs=os.path.abspath(forward_run_path)
                self.initalize_forward_meteo(meteo_startday, self.total_meteo_days, run_script_path_abs)
        
        backward_meteo_flag = (self.runconfig.get('flags')['ensemble_backward_data'].lower() == 'true')
        if backward_meteo_flag:
            self.initialize_backward_meteo()
        
        
        


    




    def initalize_forward_meteo(self,meteo_startday,total_meteo_days, run_script_path_abs):
        """Prepare forward meteo/ground inputs and precompute Kz and deposition for a period."""
        groundFile = Path(run_script_path_abs)/'data'/'ground'/'LUC-glcf.bin'
        if not groundFile.is_file():
            groundPath = Path(run_script_path_abs)/'data'/'ground'
            groundPath.mkdir(parents=True, exist_ok=True)
            luc_glcf = self.polyphemus_path + '/preprocessing/ground/luc-glcf'
            subprocess.run([luc_glcf, 'config/general.cfg', 'config/luc-glcf.cfg'], cwd=run_script_path_abs, check=True)
        groundZhangFile = Path(run_script_path_abs)/'data'/'ground'/'LUC-glcf-zhang.bin'
        if not groundZhangFile.is_file():
            subprocess.run([self.polyphemus_path + '/preprocessing/ground/luc-convert', 'config/general.cfg', 'config/glcf_to_zhang.cfg'], cwd=run_script_path_abs, check=True)
        roughnessFile = Path(run_script_path_abs)/'data'/'ground'/'Roughness-glcf.bin'
        if not roughnessFile.is_file():
            subprocess.run([self.polyphemus_path + '/preprocessing/ground/roughness', 'config/general.cfg', 'config/roughness.cfg'], cwd=run_script_path_abs, check=True)
        meteoPath = Path(run_script_path_abs)/'data'/'meteo'
        meteoPath.mkdir(parents=True, exist_ok=True)
        kzTMPath = Path(run_script_path_abs)/'data'/'meteo'/'Kz_TM'
        kzTMPath.mkdir(parents=True, exist_ok=True)
        depPath = Path(run_script_path_abs)/'data'/'dep'
        depPath.mkdir(parents=True, exist_ok=True)

        resultsPath = Path(run_script_path_abs)/'results'
        resultsPath.mkdir(parents=True, exist_ok=True)
        for daysN in range(0,total_meteo_days):
            durations = timedelta(days=daysN)
            currentDay = meteo_startday + durations
            nextDay = currentDay + timedelta(days=1)
            print('Current Day:' + currentDay.isoformat())

            subprocess.run([self.polyphemus_path + '/preprocessing/meteo/meteo_parallel', 'config/general.cfg', 'config/meteo.cfg', currentDay.strftime('%Y-%m-%d')], cwd=run_script_path_abs, check=True)

            subprocess.run([self.polyphemus_path + '/preprocessing/meteo/Kz', 'config/general.cfg', 'config/meteo.cfg', currentDay.strftime('%Y-%m-%d')], cwd=run_script_path_abs, check=True)

            subprocess.run([self.polyphemus_path + '/preprocessing/meteo/Kz_TM', 'config/general.cfg', 'config/meteo.cfg', currentDay.strftime('%Y-%m-%d'), nextDay.strftime('%Y-%m-%d')], cwd=run_script_path_abs, check=True)

            subprocess.run([self.polyphemus_path + '/preprocessing/dep/dep_aerosol', 'config/general.cfg', 'config/dep.cfg', currentDay.strftime('%Y-%m-%d'), nextDay.strftime('%Y-%m-%d')], cwd=run_script_path_abs, check=True)

    def initialize_backward_meteo(self):
        """Prepare backward run directories and reverse necessary meteo fields."""
        dataSubFolder = 'data'
        resultSubFolder = 'results'
        subfolders = ['meteo', 'dep', 'meteo/Kz_TM']
        for i in range(self.Nensemble):
            target_backward_path = self.backward_dir / str(i)
            target_data_path = target_backward_path / dataSubFolder
            target_result_path = target_backward_path / resultSubFolder
            target_data_path.mkdir(parents=True, exist_ok=True)
            target_result_path.mkdir(parents=True, exist_ok=True)
            for subfolder in subfolders:
                (target_data_path / subfolder).mkdir(parents=True, exist_ok=True)
            ground_src_path = self.forward_dir / str(i) / 'data' / 'ground'
            ground_dst_path = target_data_path / 'ground'
            copy_with_overwrite(ground_src_path,ground_dst_path)
            Nt_meteo = int(self.total_meteo_days*24/int(self.runconfig.get('meteo')['Delta_t_simulation']))
            Nx = int(self.general_config.get('domain')['Nx'])
            Ny = int(self.general_config.get('domain')['Ny'])
            Nz = int(self.general_config.get('domain')['Nz'])
            sizes = [Nx,Ny,Nz,Nt_meteo]
            typeSize = 4
            for subfolder in subfolders:
                filenames = list((self.forward_dir / str(i) / dataSubFolder / subfolder).glob('*.bin'))
                for filename in filenames:
                    file_size = filename.stat().st_size
                    sizeTemp = copy.deepcopy(sizes)
                    layerN = math.floor(file_size/sizes[0]/sizes[1]/sizes[3]/typeSize)
                    sizeTemp[2] = layerN
                    residual = file_size/(math.prod(sizeTemp)*typeSize)
                    residual = residual-math.floor(residual)
                    if residual<np.finfo(np.float32).eps:
                        factorReverse = 1
                    elif abs(residual-1/sizeTemp[0])<np.finfo(np.float32).eps:
                        factorReverse = -1
                        sizeTemp[0] = sizeTemp[0]+1
                    elif abs(residual-1/sizeTemp[1])<np.finfo(np.float32).eps:
                        factorReverse = -1
                        sizeTemp[1] = sizeTemp[1]+1
                    else:
                        print(f"Error: size error of {filename.name}")
                        sys.exit(1)
                    sizeTemp = list(map(int,sizeTemp))
                    datatmp=read_data(filename, sizeTemp)
                    datatmp=datatmp[:,:,:,::-1]
                    if factorReverse == -1:
                        datatmp= datatmp*factorReverse
                    target_File_name = target_data_path/subfolder/str(filename.name)
                    write_data(target_File_name,datatmp)





    



    
    
    def run_single_forward(self,args):
        """Run forward model for a single ensemble member and archive results."""
        i, forward_dir, polyphemus_path, current_time_step = args
        run_path = forward_dir / str(i)
        
        
        subprocess.run(
            [polyphemus_path + '/processing/photochemistry/polair3d', 'config/case.cfg'],
            cwd=run_path,
            check=True,
            stdout=subprocess.DEVNULL
        )
        
        src_dir = forward_dir / str(i) / 'results'
        tar_dir = forward_dir / str(i) / f'forward_results{current_time_step}'
        shutil.copytree(src_dir, tar_dir, dirs_exist_ok=True)
    
    def run_forward(self):
        """Run all ensemble forward simulations in parallel and archive outputs."""
        args_list = [
            (i, self.forward_dir, self.polyphemus_path, self.current_time_step)
            for i in range(self.Nensemble)
        ]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: 
            pool.map(self.run_single_forward, args_list)
    
    def initialize_keyarea(self):
        """Load key-area configuration and instantiate key-area objects."""
        keyarea_path = self.runconfig.get('path')['key_area_config']
        keyarea_config = ConfigParser()
        keyarea_config.parse(keyarea_path)
        self.keyarealist = []
        for section_dict in keyarea_config.get_section():
            from src.keyareaclass import keyarea  
            area_obj = keyarea(section_dict)
            self.keyarealist.append(area_obj)
    
    def get_keyarea_list(self):
        """Return the list of initialized key-area objects."""
        if self.keyarealist is None:
            print('The keyarealist has not been initialized.', file=sys.stderr)
            sys.exit(1)
        return self.keyarealist
    
    def update_keyarealist(self):
        """Refresh per-area time series, mean concentrations, and arrival times."""
        cal_limitvalue = False
        if self.limitvalue == None:
                if not self.effective_value_flag:
                    cal_limitvalue = True
        if cal_limitvalue == True:
            meanconc = []
            for item in self.keyarealist:
                item.cal_conclist(self.Nensemble, self.forward_dir,self.domain_size,self.current_result_size, self.forward_conc_filename)
                item.cal_meanconc()
                
                meanconc.append(item.get_meanconc())
            meanconc = np.array(meanconc)
            flattened = meanconc.flatten()  
            second_min = np.partition(flattened, 1)[1]  
            self.limitvalue = 0.3*second_min
            for item in self.keyarealist:
                item.cal_arrivetime(self.limitvalue, self.simulation_date_min,self.sim_save_delta_t,self.current_time_step)
                
        
        else:
            for item in self.keyarealist:
                item.cal_conclist(self.Nensemble, self.forward_dir,self.domain_size,self.current_result_size, self.forward_conc_filename)
                item.cal_meanconc()
                item.cal_arrivetime(self.limitvalue, self.simulation_date_min,self.sim_save_delta_t,self.current_time_step)

                

    def update_output_keyarealist(self):
        """Compute output-oriented per-area time series and arrival times."""
        for item in self.keyarealist:
            item.cal_conclist_for_output(self.Nensemble, self.forward_dir,self.domain_size,self.current_result_size, self.forward_conc_filename)
            item.cal_meanconc_for_output()
            item.cal_arrivetime_for_output(self.limitvalue, self.simulation_date_min,self.sim_save_delta_t,self.current_time_step)
    
    
    def get_keyarealist(self):
        return self.keyarealist           

    def print_keyarea(self):
        """Print basic statistics for each key area (for debugging)."""
        for item in self.keyarealist:
            name = item.get_areaname()
            print(name)
            meanconc = item.get_meanconc()
            print(meanconc)
            arrivetime = item.get_arrivetime()
            print(arrivetime)
    

    


    def run_backward(self):
        """Execute backward runs for all key areas and ensemble members, with postprocessing."""
        print(f'Running backward calculation...time:{self.current_time_step} ')
        
        for n in range(self.Nensemble):
            result_folder_path = self.backward_dir / str(n) / ('backward_results' + str(self.current_time_step))
            result_folder_path.mkdir(parents=True, exist_ok=True)

        
        for item in self.keyarealist:
            with multiprocessing.Pool(processes=self.Nensemble) as pool:
                pool.starmap(self._run_single_member, [(self, n, item) for n in range(self.Nensemble)])

        
        with multiprocessing.Pool(processes=self.Nensemble) as pool:
            pool.starmap(self._reverse_single_result, [(self, n) for n in range(self.Nensemble)])

        
        for n in range(self.Nensemble):
            src_dir = self.backward_dir / str(n) / 'results'
            tar_dir = self.backward_dir / str(n) / ('backward_results' + str(self.current_time_step))
            shutil.copytree(src_dir, tar_dir, dirs_exist_ok=True)


    @staticmethod
    def _run_single_member(self, n, item):
        """Worker to run a single ensemble backward calculation for one key area."""
        areaname = item.get_areaname()
        abscissa = item.get_abscissa()
        ordinate = item.get_ordinate()
        altitude = item.get_altitude()
        arrive_time = item.get_arrivetime()
        this_arrive_time = arrive_time[n]

        if self.backward_model == 1:
            release_begin = (self.simulation_date_max - (this_arrive_time + timedelta(hours=self.slice_h))) + self.simulation_date_min
            release_end = release_begin + timedelta(hours=self.slice_h)
        elif self.backward_model == 2:
            release_begin = self.simulation_date_min
            release_end = self.simulation_date_max - timedelta(hours=(self.current_time_step+1))  # ensure the last time step is not included

        back_config_path = self.backward_dir / str(n) / 'config'
        if self.backward_model == 2:
            conc_list = item.get_conclist()[n]
            back_release_list = conc_list[::-1]
            back_release_list.astype(np.float32).tofile(back_config_path / f'factor_Cs137_30min_{n}.bin')

        source_cfg = ConfigParser()
        source_cfg.parse(back_config_path / 'source.dat')
        source_cfg.set('source', 'Abscissa', abscissa)
        source_cfg.set('source', 'Ordinate', ordinate)
        source_cfg.set('source', 'Altitude', altitude)
        if self.backward_model == 1:
            source_cfg.set('source', 'Type', 'continuous')
            source_cfg.set('source', 'TemporalFactor', '---')
            source_cfg.set('source', 'Date_min_file', '---')
            source_cfg.set('source', 'Delta_t', '---')
        elif self.backward_model == 2:
            source_cfg.set('source', 'Delta_t', f'{self.slice_h * 3600}')
            source_cfg.set('source', 'TemporalFactor', f'config/factor_Cs137_30min_{n}.bin')
            source_cfg.set('source', 'Rate', '1e15')
        source_cfg.set('source', 'Date_beg', release_begin.strftime('%Y-%m-%d_%H-%M-%S'))
        source_cfg.set('source', 'Date_end', release_end.strftime('%Y-%m-%d_%H-%M-%S'))
        source_cfg.save(back_config_path / 'source.dat')

        saver_cfg = ConfigParser()
        saver_cfg.parse(back_config_path / 'case-saver.cfg')
        saver_cfg.set('save', 'model', areaname, 1)
        saver_cfg.save(back_config_path / 'case-saver.cfg')

        run_path = self.backward_dir / str(n)
        subprocess.run([self.polyphemus_path + '/processing/photochemistry/polair3d', 'config/case.cfg'], cwd=run_path, check=True, stdout=subprocess.DEVNULL)
        return n


    @staticmethod
    def _reverse_single_result(self, n):
        result_path = self.backward_dir / str(n) / 'results'
        result_files = list(result_path.glob('*.bin'))
        for result_file in result_files:
            filesize = result_file.stat().st_size
            if abs(filesize - 4* self.origin_result_size[0] * self.origin_result_size[1] * self.origin_result_size[2] * self.origin_result_size[3]) < np.finfo(np.float32).eps:
                size = self.origin_result_size
            else:
                size = [self.origin_result_size[0], self.origin_result_size[1], 1, self.origin_result_size[3]]
            datatemp = read_data(result_file, size)
            datatemp = datatemp[:, :, :, ::-1]
            write_data(result_file, datatemp)
        return n

    

    def set_step(self, step):
        self.current_time_step = step

    def get_domain_size(self):
        return self.domain_size
    def get_forward_dir(self):
        return self.forward_dir
    def get_backward_dir(self):
        return self.backward_dir   
    def get_Nensemble(self):
        return self.Nensemble        
    def get_origin_result_size(self):
        return self.origin_result_size
    
    def get_current_result_size(self):
        return self.current_result_size
    def get_current_size(self):
        return self.current_result_size
    def get_release_name(self):
        return self.release_name
    def get_forward_name(self):
        return self.forward_name
    
    def get_limited_value(self):
        if self.limitvalue is None:
            print('The limited value list has not been initialized.', file=sys.stderr)
            sys.exit(1)
        return self.limitvalue
    
    def get_dump_path(self):
        if self.restart_dir is None:
            print('The restart_dir has not been initialized.', file=sys.stderr)
            sys.exit(1)
        return self.restart_dir
    
    def restart_config_update(self):
        restart_t_min = (self.simulation_date_min+(self.current_time_step+1)*timedelta(hours=1)).strftime('%Y-%m-%d_%H-%M-%S')
        restart_Nt = self.simulation_nt-(self.current_time_step+1)*int(self.runconfig.get('simulation_domain')['Interval_length'])
        self.main_config.set('domain','Date_min', restart_t_min)
        self.main_config.set('domain','Nt',restart_Nt)
        self.main_config.set('options','With_initial_condition','yes')
        
        
        for i in range(self.Nensemble):
            initial_conc_file = self.restart_dir.absolute() / str(self.current_time_step) / 'ana_ensemble' / f'ensemble_{i:02d}.bin'
            
            target_config_path = self.forward_dir / str(i) / 'config'
            
            ensemble_data_config = ConfigParser()
            ensemble_data_config.parse(target_config_path / 'case-data.cfg')
            ensemble_data_config.add_key('initial_condition','Fields',self.species)
            ensemble_data_config.add_key('initial_condition','Filename',str(initial_conc_file))
            ensemble_data_config.save(target_config_path / 'case-data.cfg')
            
            ensemble_main_config = ConfigParser()
            ensemble_main_config.parse(target_config_path / 'case.cfg')
            ensemble_main_config.set('domain','Date_min', restart_t_min)
            ensemble_main_config.set('domain','Nt',restart_Nt)
            ensemble_main_config.set('options','With_initial_condition','yes')
            ensemble_main_config.save(target_config_path / 'case.cfg')




    def update_current_result_size_after_restart(self):
        nx, ny, nz = self.origin_result_size[0], self.origin_result_size[1], self.origin_result_size[2]
        nt = self.origin_result_size[3]-(self.current_time_step+1)
        self.current_result_size = [nx,ny,nz,nt]




    def get_state(self):
        return self.state_vector

    def set_state(self, new_state):
        self.state_vector = new_state

    def get_ensemble_size(self):
        return self.Nensemble

    def get_state_size(self):
        return self.Nstate

    def get_time_steps(self):
        return 10

    def get_current_date(self):
        return f"2023-01-01T{self.current_time_step:02d}:00:00"

    def init_step(self):
        print(f"Initializing time step {self.current_time_step}")

    def analyze(self):
        print(f"Analyzing results for time step {self.current_time_step}")

    def save_state(self):
        print(f"Saving state for time step {self.current_time_step}")
    

