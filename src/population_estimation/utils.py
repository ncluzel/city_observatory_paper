import pandas as pd
import numpy as np 
import sys 
import os
import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import scipy

current_dir = os.getcwd()
module_path = os.path.abspath(os.path.join(current_dir, '../..', 'libs'))
if module_path not in sys.path:
    sys.path.append(module_path)

from SCOU_NC_Vanilla_NUTS_manual import *

def get_data_and_init_dicts(step, filepath):
    scou_dict, sub_data_dict = {}, {}
    subraw_data = pd.read_excel(filepath, sheet_name=step)
    
    parameters_list = subraw_data.iloc[1, 3:10].index.tolist()
    lods = subraw_data.iloc[1, 3:10].values.tolist()
    lods_dict = dict(zip(parameters_list, lods))
    
    subraw_data = subraw_data.loc[2:]
    subraw_data = subraw_data.drop('Unnamed: 0', axis=1)
    subraw_data.dateStart = pd.to_datetime(subraw_data.dateStart)
    
    for col in parameters_list + ['plantVolume']:
        subraw_data[col] = pd.to_numeric(subraw_data[col])    

    return subraw_data, scou_dict, sub_data_dict, lods_dict, parameters_list

def append_and_process_results(input_data, this_molecule, scou, common_flux_data):
    input_data['muX'] = scou.muX
    input_data['ICL'] = scou.CIL
    input_data['ICU'] = scou.CIU
    input_data['pout'] = scou.pointwise_pout
    
    input_data.obs = np.log10(np.exp(input_data.obs))
    input_data.muX = np.log10(np.exp(input_data.muX))
    input_data.ICL = np.log10(np.exp(input_data.ICL))
    input_data.ICU = np.log10(np.exp(input_data.ICU))
    common_flux_data[this_molecule] = input_data.copy() 

    return input_data

def visualize_results(input_data, this_molecule):
    
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
        ax_dict['A'].plot(input_data.dateStart.values, input_data.muX.values, label='$X^*_{t_{opt}}$', color='green', linewidth=8, zorder=3)
        
        ax_dict['A'].plot(input_data.dateStart.values, input_data.ICL.values, label='95% CI', linestyle='--', color='green', linewidth=5, zorder=1)
        ax_dict['A'].plot(input_data.dateStart.values, input_data.ICU.values, linestyle='--', color='green', linewidth=5, zorder=1)
        
        ax_dict['A'].fill_between(input_data.dateStart.values, input_data.ICL.values, input_data.ICU.values, alpha=.3, color='green')
    
        
        scatter_points = ax_dict['A'].scatter(input_data.dateStart.values, input_data.obs.values, label='$\hat{X}_t$', 
                             c=input_data.pout.values,
                             cmap='bwr', edgecolor='black', s=360, zorder=3,
                             linewidths=1.5, alpha=0.9, vmin=0, vmax=1)
    
        
        ax_dict['A'].set_ylabel("Concentration (UG/L) - $\log_{10}$ scale")
        ax_dict['A'].set_xlabel("Sampling date")
        ax_dict['A'].tick_params(axis='x', labelsize=TICK_SIZE)
        ax_dict['A'].tick_params(axis='y', labelsize=TICK_SIZE)
        ax_dict['A'].grid(linewidth=1, color='black', alpha=0.8)
        ax_dict['A'].set_title(this_molecule, size=TITLE_SIZE)
    
        ### Outlier probability legend:
        cmin, cmax = 0.0, 1.0 
        axins1 = inset_axes(ax_dict['A'], width='2%', height='100%', loc='right', borderpad=0)
        axins1.grid(False)
        cbar = fig.colorbar(scatter_points, cax=axins1, orientation='vertical')
        
        # Setting tick limits:
        cbar.set_ticks([cmin, cmax])
        cbar.ax.yaxis.set_major_locator(ticker.FixedLocator([cmin, cmax]))
        
        # Standardizing the float format displayed:
        decimal_places = 1
        cbar.ax.set_yticklabels([f"{cmin:.{decimal_places}f}", f"{cmax:.{decimal_places}f}"], fontsize=TICK_SIZE)
    
        # Placing the label at the right spot:
        cbar.ax.set_ylabel('Outlier probability', size=TICK_SIZE, labelpad=-35)
    
        # Main legend
        plt.rcParams['text.usetex'] = False
        h1, l1 = ax_dict['A'].get_legend_handles_labels()
        fig.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=5)
        
        plt.show()

def get_sub_data_conc(input_data, lods_dict, indicator, wwtp, remove_outliers=False):
    this_lod = lods_dict[indicator]
    
    output_data = input_data.copy()
    output_data = output_data.loc[::, ['dateStart', indicator, 'plantVolume']]
    output_data.columns = ['dateStart', 'obs_conc', 'plantVolume']
    output_data['lod_conc'] = this_lod
    
    # for error checking purpose
    output_data['obs_raw'] = output_data.obs_conc.copy()
    output_data['lod_raw'] = output_data.lod_conc.copy()
    
    output_data['obs'] = output_data.obs_conc 
    output_data['lod'] = output_data.lod_conc 
    
    output_data.obs = np.log(output_data.obs)
    output_data.lod = np.log(output_data.lod)

    # Mandatory for pixel perfect reproducibility because of float precision during dataset export:
    if wwtp=='MAV' and indicator=='NGL':
        output_data.loc[output_data.dateStart=='2021-07-04', 'obs'] = 3.6648428762856953
        output_data.loc[output_data.dateStart=='2021-07-23', 'obs'] = 4.028916756899645
    
    columns = ['dateStart', 'obs', 'lod', 'plantVolume', 'obs_raw', 'lod_raw'] 
    output_data = output_data.loc[::, columns]
    output_data = output_data.set_index('dateStart').resample('D').mean().reset_index()  

    return output_data

def get_sub_data_flow(input_data, lods_dict, indicator, wwtp, remove_outliers=True, outlier_threshold=0.7):

    folder_path = '../../outputs/files/conc_data'
    this_lod = lods_dict[indicator]
    
    output_data = input_data.copy()
    output_data = output_data.loc[::, ['dateStart', indicator, 'plantVolume']]
    output_data.columns = ['dateStart', 'obs_conc', 'plantVolume']
    output_data['lod_conc'] = this_lod
    
    # for error checking purpose
    output_data['obs_raw'] = output_data.obs_conc.copy()
    output_data['lod_raw'] = output_data.lod_conc.copy()
    
    output_data['obs'] = output_data.obs_conc * output_data.plantVolume
    output_data['lod'] = output_data.lod_conc * output_data.plantVolume 
    
    output_data.obs = np.log(output_data.obs)
    output_data.lod = np.log(output_data.lod)
    
    columns = ['dateStart', 'obs', 'lod', 'plantVolume', 'obs_conc', 'lod_conc', 'obs_raw', 'lod_raw'] 
    output_data = output_data.loc[::, columns]

    if remove_outliers:
        conc_output_file = pd.read_csv(f'{folder_path}/{wwtp}/{indicator}.csv', sep=";")
        remove_those = conc_output_file.loc[conc_output_file.pout>=outlier_threshold].dateStart.tolist()
        print(f'{len(remove_those)} outliers removed.')

        print(f'Data shape before outliers removal: {output_data.shape}.')
        dropems = output_data.loc[output_data.dateStart.isin(remove_those)].index.tolist()
        output_data = output_data.drop(dropems, axis=0)
        output_data.reset_index(inplace=True, drop=True)
        print(f'Data shape after outliers removal: {output_data.shape}.')
    
    # Mandatory for pixel perfect reproducibility because of float precision during dataset export:
    if wwtp=='MAV' and indicator=='NGL':
        output_data.loc[output_data.dateStart=='2021-07-03', 'obs'] = 15.169607294210682

    elif wwtp=='CLICHY' and indicator=='DCO':
        output_data.loc[output_data.dateStart=='2021-07-28', 'obs'] = 19.317423567686664

    elif wwtp=='CLICHY' and indicator=='PT':
        output_data.loc[output_data.dateStart=='2021-07-28', 'obs'] = 14.826632510036568
    
    output_data = output_data.set_index('dateStart').resample('D').mean().reset_index()  

    return output_data

def run_MCMC(subraw_data, lods_dict, scou_dict, parameters_list, step, get_sub_data, remove_outliers, tuning_iters, sampling_iters, nb_chains):
    
    for this_parameter in parameters_list:
        print("==========")
        print(f'Processing {this_parameter}...')
        print("==========")
        
        sub_data = get_sub_data(subraw_data, lods_dict, this_parameter, step, remove_outliers)
    
        lod_vect = sub_data.lod.values
        observation_matrix = sub_data.obs.values
        
        if get_sub_data==get_sub_data_conc:
            p_out_frozen=-1
        else:
            p_out_frozen = 0.0

        scou = SCOU_RW1_NUTS(observation_matrix, lod_vect, tuning_iters=tuning_iters, sampling_iters=sampling_iters,
                                     export_name=None, 
                                     p_out_frozen=p_out_frozen, nb_chains=nb_chains, export_chains=False,
                                     RW_order=1)

        scou.fit()
    
        scou_dict[this_parameter] = scou

def select_MCMC_chains(scou_dict, sub_data_dict, parameters_list, chain_selector_dict, common_flux_data, nb_chains):

    for this_parameter in parameters_list:

        print("==========")
        print(f'Processing {this_parameter}...')
        print("==========")
            
        scou = scou_dict[this_parameter]
        sub_data = sub_data_dict[this_parameter]

        selected_chains = np.arange(nb_chains).tolist()
        remove_those = chain_selector_dict[this_parameter]
        for i in remove_those:
            selected_chains.remove(i)
        
        scou.visualize_latents(selected_chains)
        scou.predict(selected_chains)

        sub_data = append_and_process_results(sub_data, this_parameter, scou, common_flux_data)
        visualize_results(sub_data, this_parameter)

def format_func(value, tick_number):
    return f'${value:.2f}$'

def format_func_2f(value, tick_number):
    return f'${value:.2f}$'

def format_func_1f(value, tick_number):
    return f'${value:.1f}$'

def format_func_0f(value, tick_number):
    return f'${value:.0f}$'

def ape(target, pred):
    num = pred - target
    return np.abs(num / target)

def plot_boxplot(ax_dict, values, subplot_letter, color, hatch, position, legend, params_dict):
    output_boxplot = ax_dict[subplot_letter].boxplot(values, 
                                   positions=[position],
                                   whis=params_dict['whiskers'],
                                   patch_artist=params_dict['patch_artist'],
                                   medianprops=params_dict['medianprops'],
                                   flierprops=params_dict['flierprops'],
                                   whiskerprops=params_dict['whiskerprops'],
                                   capprops=params_dict['capprops'],
                                   showfliers=params_dict['showfliers'],
                                   widths=params_dict['width']
                                   )

    for index, bplot in enumerate(output_boxplot['boxes']):
        bplot.set_label(legend)
        bplot.set_facecolor(color)
        bplot.set(hatch=hatch, linewidth=3)

def plot_boxplots_block(ax_dict, key, mav_apes, sev_apes, params_dict):
    ape_mav, ape_mav_1, ape_mav_2, ape_mav_3 = mav_apes
    ape_sev, ape_sev_1, ape_sev_2, ape_sev_3 = sev_apes
    
    plot_boxplot(ax_dict, 100*ape_mav, key, color='orange', hatch='/', position=0.1, legend='Whole year', params_dict=params_dict)
    plot_boxplot(ax_dict, 100*ape_mav_1, key, color='darkorchid', hatch='/', position=-0.1, legend='Before first lockdown', params_dict=params_dict)
    plot_boxplot(ax_dict, 100*ape_mav_2, key, color='dodgerblue', hatch='/', position=-0.3, legend='During lockdowns', params_dict=params_dict)
    plot_boxplot(ax_dict, 100*ape_mav_3, key, color='forestgreen', hatch='/', position=0.3, legend='In between lockdowns', params_dict=params_dict)

    plot_boxplot(ax_dict, 100*ape_sev, key, color='orange', hatch='/', position=1+0.1, legend='Whole year', params_dict=params_dict)
    plot_boxplot(ax_dict, 100*ape_sev_1, key, color='darkorchid', hatch='/', position=1-0.1, legend='Before first lockdown', params_dict=params_dict)
    plot_boxplot(ax_dict, 100*ape_sev_2, key, color='dodgerblue', hatch='/', position=1-0.3, legend='During lockdowns', params_dict=params_dict)
    plot_boxplot(ax_dict, 100*ape_sev_3, key, color='forestgreen', hatch='/', position=1+0.3, legend='In between lockdowns', params_dict=params_dict)

def plot_stat_signif(ax_dict, subplot_letter, sig_symbol, x1, x2, LABEL_SIZE, y_top=None, y_bottom=None, bar_spacing=0.004, j=1):
    # Plotting statistical significance:
    plt.rcParams['text.usetex'] = False
    if y_top==None:
        y_bottom, y_top = ax_dict[subplot_letter].get_ylim()
    y_range = y_top - y_bottom

    y_starting_pos = (0.7 + j * 0.05)*y_top
    y_margin = 0.0125 * y_range
    
    bar_height, bar_tips = y_starting_pos + bar_spacing, y_starting_pos - y_margin + bar_spacing
    text_height = bar_height 

    ax_dict[subplot_letter].plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=3, c='k')

    ax_dict[subplot_letter].text((x1 + x2) * 0.5,
                      text_height,
                      sig_symbol,
                      ha='center',
                      va='bottom',
                      c='k',
                      fontsize=LABEL_SIZE/2)    

    return y_top, y_bottom

def plot_stat_signif_block(ax_dict, key, symbols_dict, LABEL_SIZE):
    y_tops, y_bottoms = [], []
    j=0
    # MAV
    # blue
    y_top, y_bottom = plot_stat_signif(ax_dict, key, symbols_dict['MAV - During vs before'] , -0.3, -0.1, LABEL_SIZE, bar_spacing=0.004, j=j)
    y_tops.append(y_top)
    y_bottoms.append(y_bottom)
    j+=1
    # orchid
    y_top, y_bottom = plot_stat_signif(ax_dict, key, symbols_dict['MAV - Whole vs before'], -0.1, 0.1, LABEL_SIZE, bar_spacing=0.004, j=j)
    y_tops.append(y_top)
    y_bottoms.append(y_bottom)
    j+=1
    # orange
    y_top, y_bottom = plot_stat_signif(ax_dict, key, symbols_dict['MAV - Whole vs between'], 0.1, 0.3, LABEL_SIZE, bar_spacing=0.004, j=j)
    y_tops.append(y_top)
    y_bottoms.append(y_bottom)
    j+=1
    y_top, y_bottom = plot_stat_signif(ax_dict, key, symbols_dict['MAV - During vs whole'], -0.3, 0.1, LABEL_SIZE, bar_spacing=0.004, j=j)
    y_tops.append(y_top)
    y_bottoms.append(y_bottom)
    j+=1
    y_top, y_bottom = plot_stat_signif(ax_dict, key, symbols_dict['MAV - Before vs between'], -0.1, 0.3, LABEL_SIZE, bar_spacing=0.004, j=j)
    y_tops.append(y_top)
    y_bottoms.append(y_bottom)
    j+=1
    y_top, y_bottom = plot_stat_signif(ax_dict, key, symbols_dict['MAV - During vs between'], -0.3, 0.3, LABEL_SIZE, bar_spacing=0.004, j=j)   
    y_tops.append(y_top)
    y_bottoms.append(y_bottom)
    
    # SEV
    # blue
    j=0
    _, _ = plot_stat_signif(ax_dict, key, symbols_dict['SEV - During vs before'], 1-0.3, 1-0.1, LABEL_SIZE, y_tops[0], y_bottoms[0], bar_spacing=0.004, j=j)
    j+=1
    # orchid
    _, _ = plot_stat_signif(ax_dict, key, symbols_dict['SEV - Whole vs before'], 1-0.1, 1+0.1, LABEL_SIZE, y_tops[1], y_bottoms[1], bar_spacing=0.004, j=j)
    j+=1
    # orange
    _, _ = plot_stat_signif(ax_dict, key, symbols_dict['SEV - Whole vs between'], 1+0.1, 1+0.3, LABEL_SIZE, y_tops[2], y_bottoms[2], bar_spacing=0.004, j=j)
    j+=1
    _, _ = plot_stat_signif(ax_dict, key, symbols_dict['SEV - During vs whole'], 1-0.3, 1+0.1, LABEL_SIZE, y_tops[3], y_bottoms[3], bar_spacing=0.004, j=j)
    j+=1
    _, _ = plot_stat_signif(ax_dict, key, symbols_dict['SEV - Before vs between'], 1-0.1, 1+0.3, LABEL_SIZE, y_tops[4], y_bottoms[4], bar_spacing=0.004, j=j)
    j+=1
    _, _ = plot_stat_signif(ax_dict, key, symbols_dict['SEV - During vs between'], 1-0.3, 1+0.3, LABEL_SIZE, y_tops[5], y_bottoms[5], bar_spacing=0.004, j=j) 

def get_symbol(pvalue):
    if pvalue >= 0.05:
        return 'n.s.'
    elif pvalue < 0.05 and pvalue >= 0.01:
        return '*'
    elif pvalue < 0.01 and pvalue >= 0.001:
        return '**'
    else:
        return '***'

def get_cd(s1, s2):
    std_1 = np.std(s1)
    std_2 = np.std(s2)
    mean_1 = np.mean(s1)
    mean_2 = np.mean(s2)
    
    N = len(s1)
    
    pooled_std = np.sqrt( ( (N-1)*std_1**2 + (N-1)*std_2**2 ) / ( N+N - 2 ) )
    
    cd = np.abs((mean_1 - mean_2) / pooled_std)
    return cd

def append_symbols_dict(sample_1, sample_2):
    cd = get_cd(sample_1, sample_2)
    p_value = scipy.stats.ttest_ind(sample_1, sample_2, equal_var=False, alternative='two-sided')[1]
    symbol = get_symbol(p_value)
    print(f'p-value:{p_value:.3f}, symbol:{symbol}, Cohen\'s d:{cd:.3f}')

    return symbol