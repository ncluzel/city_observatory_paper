import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LinearRegression
import glob
import scipy
from functools import reduce

from models_generation_utils import *

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

def main(raw_data_filepath, crass_pmmov_filepath, wwtp):
    data_cp = pd.read_csv(crass_pmmov_filepath, sep=";")
    data_cp.dateStart = pd.to_datetime(data_cp.dateStart)
    
    for col in ['obs', 'muX', 'ICL', 'ICU']:
        data_cp[col] = 10**(data_cp[col])
    
    fd, ld = data_cp.dateStart.tolist()[0], data_cp.dateStart.tolist()[-1]

    raw_data = get_raw_data(wwtp, raw_data_filepath, limit_2020=False)
    obs_matrix, obs_matrix_CIL, obs_matrix_CIU = get_obs_matrix(wwtp, limit_2020=False)
    those_idx = raw_data.loc[(raw_data.dateStart>=fd)&(raw_data.dateStart<=ld)].index.tolist()
    
    obs_matrix = obs_matrix[those_idx]
    obs_matrix_CIL = obs_matrix_CIL[those_idx]
    obs_matrix_CIU = obs_matrix_CIU[those_idx]
    raw_data = raw_data.loc[those_idx]
    
    remove_those = raw_data.loc[raw_data.DCO.isna()|(raw_data.plantVolume.isna())].index.tolist() # necessary for vn et al.'s model
    
    raw_obs_matrix = raw_data.loc[::, ['NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES', 'plantVolume']]
    for k in range(7):
        raw_obs_matrix.iloc[:, k] *= raw_obs_matrix.iloc[:, -1]

    raw_obs_matrix.drop(remove_those, axis=0, inplace=True)
    raw_obs_matrix = raw_obs_matrix.values
    raw_obs_matrix = raw_obs_matrix[:,:-1]
    
    'NH4', 'DCO', 'DBO', 'NTK', 'NGL', 'PT', 'MES'
    QIj_vect = np.array([10.3, 120, 60, 12.6, 12.6, 1.2, 60])
    
    # vn et al.'s smoothed:
    those_NT_hats = []
    for component in [1, 2, 4, 5]:
        temp = model_vn_et_al(obs_matrix, QIj_vect, component)
        those_NT_hats.append(temp)
    
    those_NT_hats = np.array(those_NT_hats)
    this_NT_hat_vn = those_NT_hats.mean(axis=0)    
    raw_data['Nt_hat_vn_smoothed'] = this_NT_hat_vn
    
    # vn et al.'s original:
    those_NT_hats = []
    for component in [1, 2, 4, 5]:
        temp = model_vn_et_al(raw_obs_matrix, QIj_vect, component)
        those_NT_hats.append(temp)
    
    those_NT_hats = np.array(those_NT_hats)
    this_NT_hat_vn = those_NT_hats.mean(axis=0)
    
    df_idxes = np.setxor1d(those_idx, remove_those)
    df_NT_vn = pd.DataFrame()
    df_NT_vn['Nt_hat'] = this_NT_hat_vn
    df_NT_vn.set_index(df_idxes, inplace=True)
    
    this_NT_hat_vn_filled = []
    for row in (raw_data.index.tolist()):
        if row in remove_those:
            this_NT_hat_vn_filled.append(np.nan)
        else:
            this_NT_hat_vn_filled.append(df_NT_vn.loc[row, 'Nt_hat'])
    
    raw_data['Nt_hat_vn'] = this_NT_hat_vn_filled
    pop_file = raw_data.copy()

    # our model:
    best_model_filepath = f'../../outputs/files/models/{wwtp}/best_model.csv'
    df_final_wwtp = pd.read_csv(best_model_filepath, sep=";")
    df_final_wwtp.dateStart = pd.to_datetime(df_final_wwtp.dateStart)
    pop_file = pop_file.merge(df_final_wwtp.loc[::, ['dateStart', 'Nt_hat']], on='dateStart')
    
    s1 = data_cp.copy()
    s2 = pop_file.loc[~pop_file.Nt_hat_vn.isna()]

    common_dates = np.intersect1d(s1.dateStart.values, s2.dateStart.values)
    s1 = s1.loc[s1.dateStart.isin(common_dates)]
    s2 = s2.loc[s2.dateStart.isin(common_dates)]

    with plt.style.context(['science', 'notebook', 'grid']):
    
        LABEL_SIZE = 30
        TICK_SIZE = 30
        TITLE_SIZE = 38
        LEGEND_SIZE = 30
        DATES_SIZE = 18
        figsize = (28, 10) 
        
        plt.rc('axes', labelsize=LABEL_SIZE)
        plt.rc('xtick', labelsize=TICK_SIZE)   
        plt.rc('ytick', labelsize=TICK_SIZE)
        plt.rc('figure', titlesize=TITLE_SIZE)
        plt.rc('legend', fontsize=LEGEND_SIZE)
        plt.rcParams['text.usetex'] = True
        
        fig = plt.figure(figsize=figsize, layout="constrained")
        
        ax_dict = fig.subplot_mosaic(
            """
            A
            """
        )
        
        ### A
        ax_dict['A'].plot(pop_file.dateStart.values, pop_file.Nt_hat.values, linewidth=10, zorder=3, color='orange')
        ax_dict['A'].plot(pop_file.dateStart.values, pop_file.Nt_hat.values, color='black', linewidth=3, zorder=3)
    
        ax_dict['A'].plot(pop_file.dateStart.values, pop_file.Nt_hat_vn.values, linewidth=3, zorder=3, color='lightseagreen')
        ax_dict['A'].plot(pop_file.dateStart.values, pop_file.Nt_hat_vn_smoothed.values, linewidth=3, zorder=3, color='darkorchid')
        
        ax_cases = ax_dict['A'].twinx()
    
        ax_cases.plot(data_cp.dateStart.values, data_cp.muX.values, linewidth=10, zorder=3)
        ax_cases.plot(data_cp.dateStart.values, data_cp.muX.values, color='black', linewidth=3, zorder=3)
           
        ax_cases.scatter(data_cp.dateStart.values, data_cp.obs.values)
        ax_cases.scatter(s1.dateStart.values, s1.muX.values, color='red', s=540)
        ax_dict['A'].scatter(s2.dateStart.values, s2.Nt_hat.values, color='red', s=540)
        
        # Main legend
        plt.rcParams['text.usetex'] = False
        h1, l1 = ax_dict['A'].get_legend_handles_labels()
        h2, l2 = ax_cases.get_legend_handles_labels()
        fig.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=3)        
        plt.show()

    with plt.style.context(['science', 'notebook', 'grid']):
    
        LABEL_SIZE = 30
        TICK_SIZE = 30
        TITLE_SIZE = 38
        LEGEND_SIZE = 30
        DATES_SIZE = 18
        figsize = (28, 5) #figsize = (32, 10)
        
        plt.rc('axes', labelsize=LABEL_SIZE)
        plt.rc('xtick', labelsize=TICK_SIZE)   
        plt.rc('ytick', labelsize=TICK_SIZE)
        plt.rc('figure', titlesize=TITLE_SIZE)
        plt.rc('legend', fontsize=LEGEND_SIZE)
        plt.rcParams['text.usetex'] = True
        
        fig = plt.figure(figsize=figsize, layout="constrained")
        
        ax_dict = fig.subplot_mosaic(
            """
            ABC
            """
        )

        # A 
        lr = LinearRegression()
        lr.fit(s1.muX.values.reshape(-1,1), s2.Nt_hat.values)

        ax_dict['A'].scatter(s1.muX.values, s2.Nt_hat.values, s=360, color='orange', edgecolor='black')
        ax_dict['A'].plot(s1.muX.values, lr.predict(s1.muX.values.reshape(-1,1)), linewidth=10, color='crimson')
        ax_dict['A'].plot(s1.muX.values, lr.predict(s1.muX.values.reshape(-1,1)), linewidth=3, color='black')

        # B
        s2_temp = s2.copy()
        s2_temp = s2_temp.loc[~s2_temp.Nt_hat_vn.isna()]
        lr = LinearRegression()
        lr.fit(s2_temp.Nt_hat_vn.values.reshape(-1,1), s2_temp.Nt_hat.values)

        ax_dict['B'].scatter(s2_temp.Nt_hat_vn.values, s2_temp.Nt_hat.values, s=360, color='orange', edgecolor='black')
        ax_dict['B'].plot(s2_temp.Nt_hat_vn.values, lr.predict(s2_temp.Nt_hat_vn.values.reshape(-1,1)), linewidth=10, color='crimson')
        ax_dict['B'].plot(s2_temp.Nt_hat_vn.values, lr.predict(s2_temp.Nt_hat_vn.values.reshape(-1,1)), linewidth=3, color='black')

        # C
        lr = LinearRegression()
        lr.fit(s2.Nt_hat_vn_smoothed.values.reshape(-1,1), s2.Nt_hat.values)

        ax_dict['C'].scatter(s2.Nt_hat_vn_smoothed.values, s2.Nt_hat.values, s=360, color='orange', edgecolor='black')
        ax_dict['C'].plot(s2.Nt_hat_vn_smoothed.values, lr.predict(s2.Nt_hat_vn_smoothed.values.reshape(-1,1)), linewidth=10, color='crimson')
        ax_dict['C'].plot(s2.Nt_hat_vn_smoothed.values, lr.predict(s2.Nt_hat_vn_smoothed.values.reshape(-1,1)), linewidth=3, color='black')
        
    return data_cp, pop_file, s1, s2
    
def perform_correlation_computation(s1, s2):
    corr_our_model_phages = np.round(np.corrcoef(s1.muX.values, s2.Nt_hat.values)[0,1], 3)
    s2_temp = s2.copy()
    s2_temp = s2_temp.loc[~s2_temp.Nt_hat_vn.isna()]
    s1_temp = s1.loc[s1.dateStart.isin(s2_temp.dateStart.tolist())]
    corr_vn_phages = np.round(np.corrcoef(s1_temp.muX.values, s2_temp.Nt_hat_vn.values)[0,1], 3)

    corr_our_model_vn = np.round(np.corrcoef(s2_temp.Nt_hat.values, s2_temp.Nt_hat_vn.values)[0,1], 3)
    corr_our_model_vn_smoothed = np.round(np.corrcoef(s2.Nt_hat.values, s2.Nt_hat_vn_smoothed.values)[0,1], 3)

    return corr_our_model_phages, corr_vn_phages, corr_our_model_vn, corr_our_model_vn_smoothed
