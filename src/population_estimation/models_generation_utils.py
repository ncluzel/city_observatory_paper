import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
import sys 
import os
import scipy

current_dir = os.getcwd()
module_path = os.path.abspath(os.path.join(current_dir, '../..', 'libs'))
if module_path not in sys.path:
    sys.path.append(module_path)

from SCOU_NC_Vanilla_NUTS_manual import *

def get_month_start_days():
    return pd.date_range('2023-01-01', '2023-12-31', freq='MS').dayofyear.tolist()

def get_month_start_labels():
    return pd.date_range('2023-01-01', '2023-12-31', freq='MS').strftime('%m-%d').tolist()

def get_95CI(signal, alpha = 5.0):
    signal.index = pd.to_numeric(signal.index)

    CI95_lower = []
    CI95_upper = []

    for timestep in range(signal.shape[0]):

        # Gather the samples at time t:
        drawn_at_time_t = signal.loc[timestep].values

        # Computes the lower bound:
        lower_p = alpha / 2.0
        
        # Retrieves the observation at the lower percentile index:
        lower = np.percentile(drawn_at_time_t, lower_p)

        # Computes the upper bound:
        upper_p = (100 - alpha) + (alpha / 2.0)
        
        # Retrieves the observation at the upper percentile index:
        upper = np.percentile(drawn_at_time_t, upper_p)

        CI95_lower.append(lower)
        CI95_upper.append(upper)
        
    return CI95_lower, CI95_upper

def get_raw_data(step, filepath, limit_2020=True):

    output_data = pd.read_excel(filepath, sheet_name=step)
    parameters_list = output_data.iloc[1, 3:10].index.tolist()
    
    output_data = output_data.loc[2:]
    output_data = output_data.drop('Unnamed: 0', axis=1)
    output_data.dateStart = pd.to_datetime(output_data.dateStart)
    
    for col in parameters_list + ['plantVolume']:
        output_data[col] = pd.to_numeric(output_data[col])    

    if limit_2020:
        output_data = output_data.loc[output_data.year==2020].copy()
    output_data.reset_index(inplace=True, drop=True) 

    return output_data

def get_obs_matrix(wwtp, limit_2020=True, gather_CIs=True):

    smoothed_wwtp_dict, smoothed_wwtp_dict_CIL, smoothed_wwtp_dict_CIU = {}, {}, {}
    folder_path = f'../../outputs/files/flow_data/{wwtp}/'

    for indicator in ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']:
        file_path = f'{folder_path}{indicator}.csv'
        
        traces = pd.read_csv(file_path, sep=';')
        mean_trace, CIL, CIU = traces.muX.values, traces.ICL.values, traces.ICU.values
        smoothed_wwtp_dict[indicator] = mean_trace
        smoothed_wwtp_dict_CIL[indicator], smoothed_wwtp_dict_CIU[indicator] = CIL, CIU
    
    obs_matrix = np.array(list(smoothed_wwtp_dict.values())).T
    obs_matrix = 10**(obs_matrix)
    obs_matrix_CIL = np.array(list(smoothed_wwtp_dict_CIL.values())).T
    obs_matrix_CIL = 10**(obs_matrix_CIL)
    obs_matrix_CIU = np.array(list(smoothed_wwtp_dict_CIU.values())).T
    obs_matrix_CIU = 10**(obs_matrix_CIU)
    
    if limit_2020:
        obs_matrix = obs_matrix[:366] # keeping only the year 2020 
        obs_matrix_CIL, obs_matrix_CIU = obs_matrix_CIL[:366], obs_matrix_CIU[:366]
    
    if gather_CIs:
        return obs_matrix, obs_matrix_CIL, obs_matrix_CIU
    
    return obs_matrix

def model_1_estimation(obs_matrix, QIj_vect, combis, verbose=False):
    variables = ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']

    new_QIj_vect = QIj_vect.copy()
    NT_hat_combis = {}

    if verbose:
        print("==============")
        print(new_QIj_vect)
        print("==============")
    for c in combis:
    
        components = list(c)
        nb_components = len(components)
        
        sub_obs_matrix = obs_matrix.copy()
        sub_new_QIj_vect = new_QIj_vect.copy()
        sub_obs_matrix = sub_obs_matrix[:, components]
        sub_new_QIj_vect = sub_new_QIj_vect[components]

        if verbose:
            print("---------")
            print(components)
            print(np.array(variables)[components])
            print(sub_obs_matrix.shape, sub_new_QIj_vect.shape)
            print("---------")
    
        ####
        this_pointwise_Nt_hat = []
        x = sub_new_QIj_vect
        for t in range(sub_obs_matrix.shape[0]):
            y = sub_obs_matrix[t]
            lr = LinearRegression(fit_intercept=False)
            lr.fit(x.reshape(-1,1), y)
            this_pointwise_Nt_hat.append(lr.coef_[0])
    
        this_key = ' - '.join(list(np.array(variables)[components]))
        NT_hat_combis[this_key] = this_pointwise_Nt_hat
        ####
    return NT_hat_combis

def model_3_estimation(sub_data):
    lod_matrix = np.ones(sub_data.shape[0]) * -1000
    observation_matrix = sub_data.obs.values
    
    filename = 'discard.nc'
    tuning_iters = 4000
    sampling_iters = 2000
    nb_chains = 3
    
    scou = SCOU_RW1_NUTS(observation_matrix, lod_matrix, tuning_iters=tuning_iters, sampling_iters=sampling_iters,
                                 export_name=filename, 
                                 p_out_frozen=0.0, nb_chains=nb_chains, export_chains=False,
                                 RW_order=1)
    scou.fit()
    
    selected_chains = [0, 1, 2]
    remove_those = []
    for i in remove_those:
        selected_chains.remove(i)
    
    
    scou.visualize_latents(selected_chains)
    scou.predict(selected_chains) 

    return scou

def model_4_estimation(full_obs_matrix, obs_matrix, parameter_index, QIj_vect, combis, verbose=False):
    variables = ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']

    new_QIj_vect = []
    valeur_theorique_param = QIj_vect[parameter_index]
    
    for i in range(7):
        factor = (full_obs_matrix[:,i] / full_obs_matrix[:,parameter_index]).mean()
        new_QIj_vect.append(factor * valeur_theorique_param)
    
    new_QIj_vect = np.array(new_QIj_vect)
    new_QIj_vect
    
    NT_hat_combis = {}

    if verbose:
        print("==============")
        print(new_QIj_vect)
        print(parameter_index)
        print("==============")
    for c in combis:
    
        components = list(c)
        nb_components = len(components)
        
        sub_obs_matrix = obs_matrix.copy()
        sub_new_QIj_vect = new_QIj_vect.copy()
        sub_obs_matrix = sub_obs_matrix[:, components]
        sub_new_QIj_vect = sub_new_QIj_vect[components]

        if verbose:
            print("---------")
            print(components)
            print(np.array(variables)[components])
            print(sub_obs_matrix.shape, sub_new_QIj_vect.shape)
            print("---------")
    
        this_pointwise_Nt_hat = []    
        for t in range(sub_obs_matrix.shape[0]):
        
            def params_estimation(theta):
                this_y = sub_obs_matrix[t]
                this_alpha = sub_new_QIj_vect
            
                this_alpha_theta = this_alpha * theta[0]
                
                return np.linalg.norm(this_y - this_alpha_theta)
        
        
            res = scipy.optimize.minimize(params_estimation, x0=np.ones(1))
            this_pointwise_Nt_hat.append(res['x'][0])
    
        this_key = ' - '.join(list(np.array(variables)[components]))
        NT_hat_combis[this_key] = this_pointwise_Nt_hat

    return NT_hat_combis

def model_5_estimation(full_obs_matrix, obs_matrix, parameter_index, QIj_vect, combis, verbose=False):
    variables = ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']

    new_QIj_vect = []
    valeur_theorique_param = QIj_vect[parameter_index]
    
    for i in range(7):
        factor = (full_obs_matrix[:,i] / full_obs_matrix[:,parameter_index]).mean()
        new_QIj_vect.append(factor * valeur_theorique_param)
    
    new_QIj_vect = np.array(new_QIj_vect)
    new_QIj_vect
    
    NT_hat_combis = {}

    if verbose:
        print("==============")
        print(new_QIj_vect)
        print(parameter_index)
        print("==============")
    for c in combis:
    
        components = list(c)
        nb_components = len(components)
        
        sub_obs_matrix = obs_matrix.copy()
        sub_new_QIj_vect = new_QIj_vect.copy()
        sub_obs_matrix = sub_obs_matrix[:, components]
        sub_new_QIj_vect = sub_new_QIj_vect[components]

        if verbose:
            print("---------")
            print(components)
            print(np.array(variables)[components])
            print(sub_obs_matrix.shape, sub_new_QIj_vect.shape)
            print("---------")
    
        ####
        this_pointwise_Nt_hat = []
        
        for t in range(sub_obs_matrix.shape[0]):
        
            def params_estimation(theta):
                this_y = sub_obs_matrix[t]
                this_alpha = sub_new_QIj_vect
                this_beta = np.zeros(sub_obs_matrix.shape[1])
            
                this_alpha_theta = this_alpha * theta[0]
    
                for this_index in range(nb_components):     
                    if this_index!=parameter_index:
                        this_beta[this_index] = theta[this_index+1]

                return np.linalg.norm(this_y - this_alpha_theta - this_beta)
        
        
            res = scipy.optimize.minimize(params_estimation, x0=np.ones(sub_obs_matrix.shape[1]+1))
            this_pointwise_Nt_hat.append(res['x'][0])
    
        this_key = ' - '.join(list(np.array(variables)[components]))
        NT_hat_combis[this_key] = this_pointwise_Nt_hat
        ####
    return NT_hat_combis

def model_6_estimation(full_obs_matrix, obs_matrix, parameter_index, QIj_vect, combis, verbose=False):
    variables = ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']

    new_QIj_vect = []
    param_biblio_value = QIj_vect[parameter_index]
    
    for i in range(7):
        factor = (full_obs_matrix[:,i] / full_obs_matrix[:,parameter_index]).mean()
        new_QIj_vect.append(factor * param_biblio_value)
    new_QIj_vect = np.array(new_QIj_vect)
    
    NT_hat_combis = {}
    if verbose:
        print("==============")
        print(new_QIj_vect)
        print(parameter_index)
        print("==============")
    for c in combis:
        components = list(c)
        nb_components = len(components)
        
        sub_obs_matrix = obs_matrix.copy()
        sub_new_QIj_vect = new_QIj_vect.copy()
        sub_obs_matrix = sub_obs_matrix[:, components]
        sub_new_QIj_vect = sub_new_QIj_vect[components]
        if verbose:
            print("---------")
            print(components)
            print(np.array(variables)[components])
            print(sub_obs_matrix.shape, sub_new_QIj_vect.shape)
            print("---------")
    

        this_pointwise_Nt_hat = []        
        for t in range(sub_obs_matrix.shape[0]):
        
            def params_estimation(theta):
                this_y = sub_obs_matrix[t]
                this_alpha = sub_new_QIj_vect
                this_beta = np.zeros(sub_obs_matrix.shape[1])
            
                this_alpha_theta = this_alpha * theta[0]
    
                for this_index in range(nb_components):            
                    this_beta[this_index] = theta[this_index+1]
                
                return np.linalg.norm(this_y - this_alpha_theta - this_beta)
        
        
            res = scipy.optimize.minimize(params_estimation, x0=np.ones(sub_obs_matrix.shape[1]+1))
            this_pointwise_Nt_hat.append(res['x'][0])
    
        this_key = ' - '.join(list(np.array(variables)[components]))
        NT_hat_combis[this_key] = this_pointwise_Nt_hat

    return NT_hat_combis

def model_been_et_al(obs_matrix, QIj_vect):
    variables = ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']

    new_QIj_vect = QIj_vect.copy()
    NT_hat_combis = {}

    print("==============")
    print(new_QIj_vect)
    print("==============")

    components = [0]
    nb_components = len(components)
    
    sub_obs_matrix = obs_matrix.copy()
    sub_new_QIj_vect = new_QIj_vect.copy()
    sub_obs_matrix = sub_obs_matrix[:, components]
    sub_new_QIj_vect = sub_new_QIj_vect[components]

    print("---------")
    print(components)
    print(np.array(variables)[components])
    print(sub_obs_matrix.shape, sub_new_QIj_vect.shape)
    print("---------")

    ####
    this_pointwise_Nt_hat = []
    x = sub_new_QIj_vect
    for t in range(sub_obs_matrix.shape[0]):
        y = sub_obs_matrix[t]
        lr = LinearRegression(fit_intercept=False)
        lr.fit(x.reshape(-1,1), y)
        this_pointwise_Nt_hat.append(lr.coef_[0])

    return this_pointwise_Nt_hat

def model_vn_et_al(obs_matrix, QIj_vect, component_index):
    variables = ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']

    new_QIj_vect = QIj_vect.copy()
    NT_hat_combis = {}

    print("==============")
    print(new_QIj_vect)
    print("==============")

    components = [component_index]
    nb_components = len(components)
    
    sub_obs_matrix = obs_matrix.copy()
    sub_new_QIj_vect = new_QIj_vect.copy()
    sub_obs_matrix = sub_obs_matrix[:, components]
    sub_new_QIj_vect = sub_new_QIj_vect[components]

    print("---------")
    print(components)
    print(np.array(variables)[components])
    print(sub_obs_matrix.shape, sub_new_QIj_vect.shape)
    print("---------")

    ####
    this_pointwise_Nt_hat = []
    x = sub_new_QIj_vect
    for t in range(sub_obs_matrix.shape[0]):
        y = sub_obs_matrix[t]
        lr = LinearRegression(fit_intercept=False)
        lr.fit(x.reshape(-1,1), y)
        this_pointwise_Nt_hat.append(lr.coef_[0])

    return this_pointwise_Nt_hat

def best_model_estimation(full_obs_matrix, obs_matrix, parameter_index, components, QIj_vect, verbose=False):
    variables = ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES']

    new_QIj_vect = []
    valeur_theorique_param = QIj_vect[parameter_index]
    
    for i in range(7):
        factor = (full_obs_matrix[:,i] / full_obs_matrix[:,parameter_index]).mean()
        new_QIj_vect.append(factor * valeur_theorique_param)
    
    new_QIj_vect = np.array(new_QIj_vect)
    new_QIj_vect
    
    if verbose:
        print("==============")
        print(new_QIj_vect)
        print(parameter_index)
        print("==============")

    nb_components = len(components)
    
    sub_obs_matrix = obs_matrix.copy()
    sub_new_QIj_vect = new_QIj_vect.copy()
    sub_obs_matrix = sub_obs_matrix[:, components]
    sub_new_QIj_vect = sub_new_QIj_vect[components]

    if verbose:
        print("---------")
        print(components)
        print(np.array(variables)[components])
        print(sub_obs_matrix.shape, sub_new_QIj_vect.shape)
        print("---------")

    ####
    this_pointwise_Nt_hat = []
    
    for t in range(sub_obs_matrix.shape[0]):
    
        def params_estimation(theta):
            this_y = sub_obs_matrix[t]
            this_alpha = sub_new_QIj_vect
            this_beta = np.zeros(sub_obs_matrix.shape[1])
        
            this_alpha_theta = this_alpha * theta[0]

            for this_index in range(nb_components):     
                if this_index!=parameter_index:
                    this_beta[this_index] = theta[this_index+1]

            return np.linalg.norm(this_y - this_alpha_theta - this_beta)
    
    
        res = scipy.optimize.minimize(params_estimation, x0=np.ones(sub_obs_matrix.shape[1]))
        this_pointwise_Nt_hat.append(res['x'][0])

    return this_pointwise_Nt_hat

def export_files(model_name, model_results, raw_data, output_folder):
    if model_name in ['model_1', 'model_2']:
        submodel_index = 1
        NT_hat_combis = model_results
        for this_key in list(NT_hat_combis.keys()):
            this_model = this_key
            this_model = this_model.replace(' ', '')
        
            y_2 = pd.DataFrame()
            y_2['dateStart'] = raw_data.dateStart.tolist()
            y_2['Nt_hat'] = np.array(NT_hat_combis[this_key])

            this_folder = output_folder
            this_filename = model_name + '_' + str(submodel_index) + '_' + this_model + '.csv'
            this_filepath = this_folder + this_filename
            y_2.to_csv(this_filepath, index=False, sep=";")
    
    else:
        submodel_idx_range = 7
        for submodel_index in range(submodel_idx_range):
            NT_hat_combis = model_results[submodel_index]
            for this_key in list(NT_hat_combis.keys()):
                this_model = this_key
                this_model = this_model.replace(' ', '')
            
                y_2 = pd.DataFrame()
                y_2['dateStart'] = raw_data.dateStart.tolist()
                y_2['Nt_hat'] = np.array(NT_hat_combis[this_key])
        
                this_folder = output_folder
                this_filename = model_name + '_' + str(submodel_index) + '_' + this_model + '.csv'
                this_filepath = this_folder + this_filename
                y_2.to_csv(this_filepath, index=False, sep=";")

def get_new_QIj_vect_and_CIs(obs_matrix):
    new_QIj_vect = []
    new_QIj_vect_CIL = []
    new_QIj_vect_CIU = []
    valeur_theorique_NGL = 12.6
    
    for i in range(7):
        factor = (obs_matrix[:,i] / obs_matrix[:,4]).mean()
        new_QIj_vect.append(factor * valeur_theorique_NGL)
    
        lower_factor = np.percentile(obs_matrix[:,i] / obs_matrix[:,4], 2.5)
        upper_factor = np.percentile(obs_matrix[:,i] / obs_matrix[:,4], 97.5)
    
        new_QIj_vect_CIL.append(lower_factor * valeur_theorique_NGL)
        new_QIj_vect_CIU.append(upper_factor * valeur_theorique_NGL)
    
    new_QIj_vect = np.array(new_QIj_vect)
    new_QIj_vect_CIL = np.array(new_QIj_vect_CIL)
    new_QIj_vect_CIU = np.array(new_QIj_vect_CIU)
    
    print(f'NH4: {new_QIj_vect[0]:.1f}  [{new_QIj_vect_CIL[0]:.1f}, {new_QIj_vect_CIU[0]:.1f}]')
    print(f'COD: {new_QIj_vect[1]:.1f} [{new_QIj_vect_CIL[1]:.1f}, {new_QIj_vect_CIU[1]:.1f}]')
    print(f'BOD: {new_QIj_vect[2]:.1f}  [{new_QIj_vect_CIL[2]:.1f}, {new_QIj_vect_CIU[2]:.1f}]')
    print(f'TKN: {new_QIj_vect[3]:.1f}  [{new_QIj_vect_CIL[3]:.1f}, {new_QIj_vect_CIU[3]:.1f}]')
    print(f'TP:  {new_QIj_vect[-2]:.1f}   [{new_QIj_vect_CIL[-2]:.1f}, {new_QIj_vect_CIU[-2]:.1f}]')
    print(f'SS:  {new_QIj_vect[-1]:.1f}  [{new_QIj_vect_CIL[-1]:.1f}, {new_QIj_vect_CIU[-1]:.1f}]')