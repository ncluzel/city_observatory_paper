import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import datetime
from scipy.stats import gaussian_kde
import pandas as pd

# Fonction pour calculer la corrélation croisée
def cross_correlation_matcher(signal_1, signal_2, 
                              signal_1_name, signal_2_name,
                              signal_1_ylabel, signal_2_ylabel,
                              export_file_name,
                              lag_limit_left=-30, lag_limit_right=30,
                              inside_left_bound=-1, 
                              inside_right_bound=-1,
                              plot_signals=True,
                              subsampling=True, ndraws=1000, fraction=.1):

    # Creating subsignals from bounds:
    if inside_left_bound==-1 and inside_right_bound==-1:
        inside_left_bound = pd.to_datetime(max(signal_1.time.values[0], signal_2.time.values[0]))
        inside_right_bound = pd.to_datetime(min(signal_2.time.values[-1], signal_2.time.values[-1]))

    subsignal_1 = signal_1.loc[(signal_1.time>=inside_left_bound) & (signal_1.time<=inside_right_bound)]
    subsignal_2 = signal_2.loc[(signal_2.time>=inside_left_bound) & (signal_2.time<=inside_right_bound)]

    # Variables initialization:
       
    tested_lags = np.arange(lag_limit_left, lag_limit_right)

    if lag_limit_left==lag_limit_right and lag_limit_left==0:
        tested_lags = np.array([0]) 
        
    correlations_dict = {}
    alternate_correlations_dict = {}
    effets_de_bord = 0
    optimal_corr = -1
    optimal_lag = -1

    # Getting greater lag:
    greater_lag = np.max(np.abs(tested_lags))

    lag_specific_draws_dict = {}
    for lag in tested_lags:
        lag_specific_draws_dict[lag] = []

    full_lag_left_bound = inside_left_bound + datetime.timedelta(days=int(greater_lag))
    full_lag_right_bound = inside_right_bound - datetime.timedelta(days=int(greater_lag))

    if full_lag_right_bound < full_lag_left_bound:
        print("Invalid bounds. full_lag_right_bound is lower than full_lag_left_bound. Please reduce the maximum of the absolute value of tested lags, reduce inside_left_bound or increase inside_right_bound.")
        return

    
    if subsampling==False:
        fraction=1.0
        ndraws=1
        
    # For reproducibility:
    np.random.seed(48) 

    # Results are stored in two separate lists, depending on which signal is moved when considering a lag.
    # Moving signal 1 an amount of lag days towards the left would technically be the same thing as 
    # moving signal 2 and amount of lag days towards the right if they were constant signals.
    # In the general case, this is not exactly identical, so we assess both situations and aggregate the results in specific lists.
    signal_1_shift_corr_list = []
    signal_1_shift_lag_list = []
    signal_2_shift_corr_list = []
    signal_2_shift_lag_list = []

    aggregate_corr_list = []
    aggregate_lag_list = []
    
    for this_draw in range(ndraws):
        # We need to remove the points that are going outside the bounds for every studied lag.
        # Otherwise, we would not compute the lag on the same amount of points, which would induce bias to the results:
        subsample_subsignal_1 = subsignal_1.copy()
        subsample_subsignal_1 = subsample_subsignal_1.loc[(subsample_subsignal_1.time>=full_lag_left_bound) & (subsample_subsignal_1.time<=full_lag_right_bound)]
        subsample_subsignal_1 = subsample_subsignal_1.sample(frac=fraction).sort_values(by='time')

        subsample_subsignal_2 = subsignal_2.copy()
        subsample_subsignal_2 = subsample_subsignal_2.loc[subsample_subsignal_2.time.isin(subsample_subsignal_1.time)]

        alternate_corr_list = []
        corr_list = []
        lag_list = []

        if subsample_subsignal_1.shape[0]<2 or subsample_subsignal_2.shape[0]<2:
            print('The number of points in subsignals is less than 2. Correlation can\'t be computed. Please reduce the maximum of the absolute value of tested lags, reduce inside_left_bound or increase inside_right_bound.')
            return
        
        for lag in tested_lags:
            # Start from the whole subsignal:
            shifted_subsignal_1 = subsample_subsignal_1.copy()
            # Add required lag:
            shifted_subsignal_1.time += datetime.timedelta(days=int(lag))
    
            # Start from the whole subsignal:
            shifted_subsignal_2 = subsample_subsignal_2.copy()
            # Add required lag:
            # Lag is negative here considering that if we move subsignal 1 towards one direction, we must move 
            # subsignal 2 towards the opposite one
            shifted_subsignal_2.time += datetime.timedelta(days=int(-lag))
     
            # We now need to get the matching indexes of the other whole signal for each case:
            requested_signal_2 = subsignal_2.copy()
            requested_signal_2 = requested_signal_2.loc[requested_signal_2.time.isin(shifted_subsignal_1.time)]
            
            requested_signal_1 = subsignal_1.copy()
            requested_signal_1 = requested_signal_1.loc[requested_signal_1.time.isin(shifted_subsignal_2.time)]
            
            # Compute both correlations:  
            corr_result = np.corrcoef(shifted_subsignal_1.signal, requested_signal_2.signal)[0, 1]
            alternate_corr_result = np.corrcoef(requested_signal_1.signal, shifted_subsignal_2.signal)[0, 1]
            
            effective_draw_size = requested_signal_1.shape[0] / subsignal_1.shape[0]
            lag_specific_draws_dict[lag].append(effective_draw_size)

            lag_list.append(lag)
            corr_list.append(corr_result)
            alternate_corr_list.append(alternate_corr_result)

        max_corr = np.max(corr_list)
        max_lag = lag_list[np.argmax(corr_list)]

        max_alternate_corr = np.max(alternate_corr_list)
        max_alternate_lag = lag_list[np.argmax(alternate_corr_list)]

        signal_1_shift_corr_list.append(max_corr)
        signal_1_shift_lag_list.append(max_lag)
        signal_2_shift_corr_list.append(max_alternate_corr)
        signal_2_shift_lag_list.append(max_alternate_lag)

        agg_corr = np.median(np.array([max_corr, max_alternate_corr]))
        agg_lag = np.median(np.array([max_lag, max_alternate_lag]))
        aggregate_corr_list.append(agg_corr)
        aggregate_lag_list.append(agg_lag)
    
    estimated_lag = np.median(aggregate_lag_list)
    estimated_shifted_subsignal_1 = subsignal_1.copy()
    estimated_shifted_subsignal_1.time += datetime.timedelta(days=int(estimated_lag))
    estimated_shifted_subsignal_1 = estimated_shifted_subsignal_1.loc[(estimated_shifted_subsignal_1.time>=inside_left_bound) & (estimated_shifted_subsignal_1.time<=inside_right_bound)]

    # Signals visualization:
    if plot_signals:
        with plt.style.context(['science', 'notebook', 'grid']):
            
            LABEL_SIZE = 30
            TICK_SIZE = 30
            TITLE_SIZE = 38
            LEGEND_SIZE = 30
            DATES_SIZE = 18
            #figsize = (32, 10)
            figsize = (32, 24)
            
            plt.rc('axes', labelsize=LABEL_SIZE)
            plt.rc('xtick', labelsize=TICK_SIZE)
            plt.rc('ytick', labelsize=TICK_SIZE)
            plt.rc('figure', titlesize=TITLE_SIZE)
            plt.rc('legend', fontsize=LEGEND_SIZE)
            plt.rcParams['text.usetex'] = True
            
            fig = plt.figure(figsize=figsize, layout="constrained")
            
            ax_dict = fig.subplot_mosaic(
                """
                AAA
                BCD
                EFG
                """
            )

            # ---------------------------------------------------- Subplot A ---------------------------------------------------- #
            ax_dict['A'].plot(signal_1.time.values, signal_1.signal.values, linewidth=5, color='dodgerblue', label='Signal 1 - ' + signal_1_name)
            ax_dict['A'].plot(signal_1.time.values, signal_1.signal.values, linewidth=1, color='black')

            ax_dict['A'].plot(subsignal_1.time.values, subsignal_1.signal.values, linewidth=5, color='red', label='Subsignal 1')
            ax_dict['A'].plot(subsignal_1.time.values, subsignal_1.signal.values, linewidth=1, color='black')

            ax_dict['A'].scatter(subsample_subsignal_1.time.values, subsample_subsignal_1.signal.values, s=360, color='red', label='Subsampled signal 1', alpha=.5)

            ax_dict['A'].plot(estimated_shifted_subsignal_1.time.values, estimated_shifted_subsignal_1.signal.values, linewidth=5, color='red', label='Estimated shifted signal 1', alpha=.5)
            ax_dict['A'].plot(estimated_shifted_subsignal_1.time.values, estimated_shifted_subsignal_1.signal.values, linewidth=1, color='black', alpha=.5)
            
            ax_dict['A'].axvline(inside_left_bound, color='black', linestyle='--', linewidth=2)
            ax_dict['A'].axvspan(inside_left_bound, inside_right_bound, color='forestgreen', alpha=.3)
            ax_dict['A'].axvline(inside_right_bound, color='black', linestyle='--', linewidth=2)
    
            ax_twin = ax_dict['A'].twinx()
    
            ax_twin.plot(signal_2.time.values, signal_2.signal.values, linewidth=5, color='orange', label='Signal 2 - ' + signal_2_name)
            ax_twin.plot(signal_2.time.values, signal_2.signal.values, linewidth=1, color='black')

            ax_twin.plot(subsignal_2.time.values, subsignal_2.signal.values, linewidth=5, color='forestgreen', label='Subsignal 2')
            ax_twin.plot(subsignal_2.time.values, subsignal_2.signal.values, linewidth=1, color='black')

            ax_twin.scatter(subsample_subsignal_2.time.values, subsample_subsignal_2.signal.values, s=360, color='forestgreen', label='Subsampled signal 2', alpha=.5)
        
            #ax_dict['A'].set_title(plantName, size=TITLE_SIZE)
            ax_dict['A'].set_ylabel(signal_1_ylabel)
            ax_twin.set_ylabel(signal_2_ylabel)
            ax_dict['A'].set_xlabel("Sampling date")
            ax_dict['A'].tick_params(axis='x', labelsize=TICK_SIZE)
            ax_dict['A'].tick_params(axis='y', labelsize=TICK_SIZE)
    
            h1, l1 = ax_dict['A'].get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            fig.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=4)

            # ---------------------------------------------------- Subplot A ---------------------------------------------------- #
            colors_key_dict = {}
            colors_key_dict['B'] = 'royalblue'
            colors_key_dict['C'] = 'forestgreen'
            colors_key_dict['D'] = 'orange'
            colors_key_dict['E'] = 'gold'
            colors_key_dict['F'] = 'gray'
            colors_key_dict['G'] = 'orangered'

            ylabels_dict = {}
            ylabels_dict['B'] = 'S1 shifted correlation'
            ylabels_dict['C'] = 'Aggregated correlation'
            ylabels_dict['D'] = 'S2 shifted correlation'
            ylabels_dict['E'] = 'S1 shifted lag'
            ylabels_dict['F'] = 'Aggregated lag'
            ylabels_dict['G'] = 'S2 shifted lag'

            distribution_key_dict = {}
            distribution_key_dict['B'] = np.array(signal_1_shift_corr_list)
            distribution_key_dict['C'] = np.array(aggregate_corr_list)
            distribution_key_dict['D'] = np.array(signal_2_shift_corr_list)
            distribution_key_dict['E'] = np.array(signal_1_shift_lag_list)
            distribution_key_dict['F'] = np.array(aggregate_lag_list)
            distribution_key_dict['G'] = np.array(signal_2_shift_lag_list)

            keylist = ['B', 'C', 'D', 'E', 'F', 'G']
            for key in keylist:       

                this_distribution = distribution_key_dict[key]
                percentile_min = np.percentile(this_distribution, 2.5)
                percentile_max = np.percentile(this_distribution, 97.5)

                #if np.mean(signal_1_shift_corr_list)<1.0:
                if np.unique(this_distribution).shape[0] > 1:
                    kde = gaussian_kde(this_distribution)
                    x_kde = np.linspace(min(this_distribution), max(this_distribution), 1000)
                    y_kde = kde(x_kde)
    
                    ax_dict[key].plot(x_kde, y_kde, linewidth=5, color=colors_key_dict[key])
                    ax_dict[key].plot(x_kde, y_kde, linewidth=1, color='black') 

        
                ax_dict[key].axvline(x=percentile_min, color=colors_key_dict[key], linewidth=5, zorder=1)
                ax_dict[key].axvline(x=percentile_max, color=colors_key_dict[key], linewidth=5, zorder=1)
                ax_dict[key].axvspan(xmin=percentile_min, xmax=percentile_max, color=colors_key_dict[key], alpha=0.5)                
                ax_dict[key].hist(this_distribution, density=True,
                                     color=colors_key_dict[key], edgecolor='black', linewidth=6)#, bins=12)
                                     #linewidth=5)
        
                #ax_dict[key].set_title(ylabels_dict[key], size=TITLE_SIZE)
                ax_dict[key].set_ylabel(ylabels_dict[key], size=LABEL_SIZE)
                ax_dict[key].set_xlabel("Value", size=LABEL_SIZE)
                ax_dict[key].tick_params(axis='x', labelsize=TICK_SIZE)
                ax_dict[key].tick_params(axis='y', labelsize=TICK_SIZE)
                ax_dict[key].tick_params(axis='x')
            
            fig.tight_layout()
            plt.savefig(export_file_name, bbox_inches = 'tight')
            plt.show()

    return aggregate_corr_list, aggregate_lag_list
