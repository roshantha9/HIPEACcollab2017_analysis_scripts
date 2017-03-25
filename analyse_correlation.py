import sys
import csv
import json
import pprint
import matplotlib
import numpy  as np
from scipy import stats
import collections
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')

from common import target_metrics_order,reduced_target_metrics_order, \
                    reduced_metrics_onlyfreqs_order,reduced_target_metrics_order_2, \
                     load_csv, calc_and_update_cpu_util, _normalise_list, \
                     scenario_list, rename_metrics,\
                     CPU_FREQS_KHZ, GPU_FREQS_MHZ


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

NUM_CPU_CORES = 4
SAMPLING_PERIOD = 200


    
    
####################
# Data manipulation
####################



    
####################
# Stats calc
####################
def compute_correlation_matrix(dataset, metrics):     
    corr_matrix = collections.OrderedDict()
    corr_mat_2d_lst = np.empty([len(metrics), len(metrics)])
    corr_nonredundant_lst = {}
    
    for ix_i, each_metric_i in enumerate(metrics):
        corr_matrix[each_metric_i]={}
        #corr_mat_2d_lst[ix_i]=[None]*len(metrics)
        for ix_j, each_metric_j in enumerate(metrics):
            x = dataset[each_metric_i]
            y = dataset[each_metric_j]
            #print each_metric_i, each_metric_j,
            #print len(x), len(y)
            pr = stats.pearsonr(x, y)
            #print pr
            corr_matrix[each_metric_i][each_metric_j] = pr[0] if np.isnan(pr[0])==False else 0.0
            corr_mat_2d_lst[ix_i][ix_j] = pr[0] if np.isnan(pr[0])==False else 0.0
            
            if each_metric_i != each_metric_j: # omit equal metrics (diagonal line)
                k1str = ' vs.\n '.join((rename_metrics(each_metric_i), rename_metrics(each_metric_j)))
                k2str = ' vs.\n '.join((rename_metrics(each_metric_j), rename_metrics(each_metric_i)))
                if (k1str not in corr_nonredundant_lst) and (k2str not in corr_nonredundant_lst): 
                    if _corr_met_ignore_list(k1str) == False:                   
                        corr_nonredundant_lst[k1str] =  pr[0] if np.isnan(pr[0])==False else 0.0
                    else:
                        pass
                    #print each_metric_i, each_metric_j
                else:
                    pass
            else:
                pass
            
    return corr_matrix, corr_mat_2d_lst, corr_nonredundant_lst


def compute_corrmatrix_all_scenarios(sc_list, met_list):
    MIF_FREQ = "default"
    INT_FREQ = "default"
    
    all_sc_corr_mat = collections.OrderedDict()
    for each_scenario in sc_list:
        DATA_DIR = BASE_DATA_DIR + each_scenario + "/"
        
        cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
        mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)

        (count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)

        lbl = "{0}:mif-{1}:int-{2}".format(SCENARIO_ID, MIF_FREQ, INT_FREQ)

        corr = compute_correlation_matrix(perfdata, met_list)[2]
        all_sc_corr_mat[each_scenario] = corr
        
    return all_sc_corr_mat
         


####################
# Plotting
####################
def plot_cross_correlation(met1_data, met2_data, norm=True):
    a = met1_data
    v = met2_data
    
    print len(a)
    
    #a = (a - np.mean(a)) / (np.std(a) * len(a))
    #v = (v - np.mean(v)) /  np.std(v)
    
    c = np.correlate(a, v, 'full')
    fig = plt.figure(figsize=(8*1.2, 4*1.2))
    fig.canvas.set_window_title("plot_cross_correlation")
    
    print len(c)
    
    plt.plot(c)


def plot_cpugpumem_dist_all_scenarios(sc_list, mif_freq, int_freq):
    cmap = plt.get_cmap('rainbow')
    colsd = [cmap(i) for i in np.linspace(0, 1, 6)]
    
    
    colsd = [
             '#08519c', '#6baed6', # blues
             '#a50f15', '#fb6a4a', # reds
             '#006d2c', '#74c476', # greens
             ]
    
    
    fig, axs = plt.subplots(5,3, figsize=(12*1.2, 10*1.2), sharex=True)    
    axs = axs.ravel()

    for ix, each_scenario in enumerate(sc_list):
        print each_scenario
        DATA_DIR = BASE_DATA_DIR + each_scenario + "/"        
        cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(mif_freq, int_freq)
        mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(mif_freq, int_freq)
        (count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)

        cpu_util = perfdata['cpu_util'] 
        gpu_util = perfdata['gpu_util']
        mem_util = perfdata['sat_total']
                
        cpu_cost = perfdata['cpu_cost'] 
        gpu_cost = perfdata['gpu_cost']
        mem_cost = perfdata['sat_cost']
        
                    
        y_data = [cpu_util, cpu_cost, 
                  gpu_util, gpu_cost,
                  mem_util, mem_cost]
        
        pos = np.arange(1, len(y_data)+1)
        xticklbls = [
                        'cpu_util', 'cpu_cost', 
                        'gpu_util', 'gpu_cost',
                        'mem_util', 'mem_cost'
                     ]
        bp = axs[ix].boxplot(y_data, positions=pos, patch_artist=True)
        # change col of boxes
        for box, c in zip(bp['boxes'], colsd):            
            box.set( facecolor =  c) # change fill color
                    
        
        axs[ix].set_xticks(pos)
        ymax =   100.0 if np.max([np.max(m) for m in y_data])<100.0 else np.max([np.max(m) for m in y_data])
                        
        axs[ix].set_ylim([-0.5, ymax])
        axs[ix].set_title(each_scenario, fontsize=16)
    
       
    axs[-1].set_xticklabels(xticklbls, rotation=35, fontsize=14)
    axs[-2].set_xticklabels(xticklbls, rotation=35, fontsize=14)
    axs[-3].set_xticklabels(xticklbls, rotation=35, fontsize=14)
    
    plt.subplots_adjust(top=0.97, left=0.03, right=.99, bottom=0.08)
    


def plot_corr_across_scenarios(all_sc_corrs):
    print(json.dumps(all_sc_corrs, indent=4))
    
    fig = plt.figure(figsize=(20*1.2, 8*1.2))
    fig.canvas.set_window_title("plot_corr_across_scenarios")
    
    # arange data
    tmpk = all_sc_corrs.keys()[0]
    corr_metric_list = all_sc_corrs[tmpk].keys()
    sorted_corr_metric_list = sorted(corr_metric_list)
    #sorted_corr_metric_list = corr_metric_list
    
    data = []
    for each_c in sorted_corr_metric_list:
        data.append([v[each_c] for k,v in all_sc_corrs.iteritems()])
        #pprint.pprint(data[-1])
       
    # plot data
    pos = np.arange(len(sorted_corr_metric_list))
    xlbls = [s.replace(',',',\n') for s in sorted_corr_metric_list]
    
    print len(data), len(pos)
    
    plt.boxplot(data, positions=pos)
    plt.xticks(pos, xlbls, rotation=35, fontsize=10)
    pprint.pprint(xlbls) 
    plt.subplots_adjust(top=0.97, left=0.03, right=0.99, bottom=0.12)
    ax = plt.gca()
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    

def plot_corr_matrix(data, metrics_order, scenario_id):
    #fig = plt.figure(figsize=(6*1.2, 6*1.2))
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('plot_corr_matrix -'+scenario_id)
    plt.pcolormesh(data, cmap='RdBu', vmin=-1.0, vmax=1.0, edgecolor='black', linestyle=':', lw=1.0)
    plt.colorbar()
    
    ticks = np.arange(0.5,len(metrics_order))
    labels = [rename_metrics(m) for m in metrics_order]
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    
    
    plt.subplots_adjust(top=0.95, left=0.16, right=1.0, bottom=0.20)

    #for axis in [ax.xaxis, ax.yaxis]:
    #    axis.set(ticks=np.arange(0.5, len(labels)), ticklabels=labels)


def plot_overlapped(perfdata, metric_list, scenario_id):
    
    fig, axs = plt.subplots(len(metric_list),1, figsize=(8*1.2, 10*1.2), sharex=True)
    fig.subplots_adjust(hspace = .1, wspace=.001)
    fig.canvas.set_window_title("plot_overlapped -" +scenario_id)
    axs = axs.ravel()
    
    samples = len(perfdata["cpu_util"])    
    total_duration = (samples * SAMPLING_PERIOD)/1000.0    
    x_data = np.linspace(0, total_duration, samples)
    
    for i in np.arange(len(metric_list)):
        #print len(perfdata[metric_list[i]]), metric_list[i]
        y_data = perfdata[metric_list[i]]
        axs[i].plot(x_data, y_data, marker='x', markeredgecolor='r')
        
        axs[i].set_ylabel(rename_metrics(metric_list[i]))
        plt.setp(axs[i].get_xticklabels(), visible=False)
        axs[i].tick_params(labelsize=10)
        
        y_min = -0.75 if _is_metric_percentage(metric_list[i]) else np.min(y_data)*0.05
        y_max = 100.75 if _is_metric_percentage(metric_list[i]) else np.max(y_data)*1.05
        axs[i].set_ylim([y_min, y_max])
        axs[i].grid(True, which='both')
        #axs[i].tick_params(axis='x',which='both')
        #axs[i].minorticks_on()
        
    plt.setp(axs[-1].get_xticklabels(), visible=True)
    axs[-1].set_xlabel("time (s)")
    plt.subplots_adjust(top=0.99, left=0.13, right=.97, bottom=0.05)


def _is_metric_percentage(m):
    r = True if ("sat_" in m) or ("util" in m) or ("cost" in m) else False
    return r 


def _corr_met_ignore_list(m):
    r = False
    if (m == "gpu_cost vs.\n gpu_freq") or (m == "gpu_freq vs.\n gpu_cost"): r=True
    
    
    if (m == "cpu_cost vs.\n cpu_freq") or (m == "cpu_freq vs.\n cpu_cost"): r=True
    
    if (m == "cpu_util vs.\n cpu_cost") or (m == "cpu_cost vs.\n cpu_util"): r=True
    if (m == "bus_freq(INT) vs.\n bus_cost") or (m == "bus_cost vs.\n bus_freq(INT)") : r=True
    if (m == "bus_freq(MIF) vs.\n bus_cost") or (m == "bus_cost vs.\n bus_freq(MIF)") : r=True
    if (m == "bus_util vs.\n bus_cost") or (m == "bus_cost vs.\n bus_util") : r=True
    
    return r


#################
#    MAIN code
#################
SCENARIO_ID = "rlbench_mprndmemi" 
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
MIF_FREQ = "default"
INT_FREQ = "default"
#MIF_FREQ = 400000
#INT_FREQ = 50000

cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)

(count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)

# plot_cross_correlation(perfdata['bus_mif_freq'],
#                        perfdata['cpu_util_freq'])


lbl = "{0}:mif-{1}:int-{2}".format(SCENARIO_ID, MIF_FREQ, INT_FREQ)

#corr_matrix = compute_correlation_matrix(perfdata, reduced_target_metrics_order_2)[1]
#plot_corr_matrix(corr_matrix, reduced_target_metrics_order_2, SCENARIO_ID)

plot_overlapped(perfdata, ['cpu_freq', 'gpu_freq',
                           'bus_mif_freq', 'bus_int_freq', 
                           'cpu_util', 'cpu_cost', 'gpu_util', 'gpu_cost',
                           'sat_total', 'sat_cost'], 
                SCENARIO_ID)

#all_sc_corrs = compute_corrmatrix_all_scenarios(scenario_list, reduced_target_metrics_order)
#plot_corr_across_scenarios(all_sc_corrs)
   
#plot_cpugpumem_dist_all_scenarios(scenario_list, "default", "default")

plt.show()
print "-- Finished --"
