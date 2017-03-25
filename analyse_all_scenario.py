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

import matplotlib.patches as patches

plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')

from common import target_metrics_order,reduced_target_metrics_order, \
                    reduced_metrics_onlyfreqs_order,reduced_target_metrics_order_2, \
                     load_csv, calc_and_update_cpu_util, _normalise_list, \
                     scenario_list, rename_metrics,\
                     CPU_FREQS_KHZ, GPU_FREQS_MHZ, INT_FREQS_KHZ, MIF_FREQS_KHZ


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

NUM_CPU_CORES = 4
SAMPLING_PERIOD = 200


    
    
####################
# Data manipulation
####################



    
####################
# Stats calc
####################
def plot_freq_time_in_state(sc_list, metric, TMP_MIF_FREQ = "default", TMP_INT_FREQ = "default"):
    f, axarr = plt.subplots(2,8, sharex=True, sharey=True, figsize=(16, 10))
    f.canvas.set_window_title('plot_time_in_state -'+metric)
    axarr = axarr.ravel()
    
    # cols 
    cmap = plt.get_cmap('autumn')
    freq_list = []
    if metric == 'cpu_freq' :        
        colsd = {f: cmap(i) for (f, i) in zip(CPU_FREQS_KHZ, np.linspace(0, 1, len(CPU_FREQS_KHZ)))}
        #cols = [cmap(i) for i in np.linspace(0, 1, len(CPU_FREQS_KHZ))]
        freq_list = CPU_FREQS_KHZ
    elif metric == 'gpu_freq' :        
        colsd = {f: cmap(i) for (f, i) in zip(GPU_FREQS_MHZ, np.linspace(0, 1, len(GPU_FREQS_MHZ)))}
        freq_list = GPU_FREQS_MHZ
    elif metric == 'bus_mif_freq' :        
        colsd = {f: cmap(i) for (f, i) in zip(MIF_FREQS_KHZ, np.linspace(0, 1, len(MIF_FREQS_KHZ)))}
        #cols = [cmap(i) for i in np.linspace(0, 1, len(MIF_FREQS_KHZ))]
        freq_list = MIF_FREQS_KHZ        
    elif metric == 'bus_int_freq' :
        colsd = {f: cmap(i) for (f, i) in zip(INT_FREQS_KHZ, np.linspace(0, 1, len(INT_FREQS_KHZ)))}        
        #cols = [cmap(i) for i in np.linspace(0, 1, len(INT_FREQS_KHZ))]
        freq_list = INT_FREQS_KHZ          
    else:
        sys.exit("Error - invalid freq type")    
        
        
    #cols = sns.color_palette("PuBuGn", n_colors=3)
    #cols = sns.cubehelix_palette(len(unique_vals))
    
    
    for ix, each_scenario in enumerate(sc_list):
        print each_scenario
        DATA_DIR = BASE_DATA_DIR + each_scenario + "/"        
        cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(TMP_MIF_FREQ, TMP_INT_FREQ)
        mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(TMP_MIF_FREQ, TMP_INT_FREQ)
        
        try:
            axarr[ix].set_title(each_scenario, fontsize=12)
            (count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)
            freq_data = perfdata[metric] 
        
            freq_data_dict = collections.OrderedDict()
            for each_freq in freq_list:
                fcount = freq_data.count(each_freq)
                if fcount > 0:            
                    freq_data_dict[each_freq] = fcount 
                    
            print(json.dumps(freq_data_dict, indent=4))
            
            cols = [colsd[f] for f in freq_data_dict.keys()]        
            pdata = freq_data_dict.values()
            
            lbls = [str(k) for k in freq_data_dict.keys()]
            wedges, plt_labels, junk = axarr[ix].pie(pdata, autopct=my_autopct, colors=cols)
            for t in plt_labels:
                    t.set_fontsize(10)                 
                    
                                
        except:
            continue

          
                                       
        axarr[ix].set_title(each_scenario, fontsize=12)
    
    rects_list=[]
    rect_lbl_list=[]
    for ix, each_f in enumerate(freq_list):
        rec = patches.Rectangle( (0.72, 0.1), 0.2, 0.6, facecolor=colsd[each_f])
        rects_list.append(rec) 
        rect_lbl_list.append(str(each_f))
    
    nlines = len(freq_list)
    ncol = int(np.ceil(nlines/2.))
    leg = plt.figlegend( rects_list, rect_lbl_list, loc = 'upper center', ncol=ncol, labelspacing=0. , fontsize=14)
    leg.draggable()
    
    plt.subplots_adjust( hspace=0.09,
                         wspace=0.00,
                         top=0.87,
                         bottom=0.0,
                         left=0.0,
                         right=1.00 )

    #plt.tight_layout()
    
    
    
def my_autopct(pct):
    return ('%1.1f%%' % pct) if pct > 4.0 else ''


####################
# Plotting
####################


def plot_cpugpumem_dist_all_scenarios(sc_list, mif_freq, int_freq):
    fig, axs = plt.subplots(5,3, figsize=(8*1.2, 8*1.2), sharex=True)
    
    axs = axs.ravel()

    for ix, each_scenario in enumerate(sc_list):
        DATA_DIR = BASE_DATA_DIR + each_scenario + "/"        
        cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(mif_freq, int_freq)
        mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(mif_freq, int_freq)
        (count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)

        cpu_util = perfdata['cpu_cost'] 
        gpu_util = perfdata['gpu_cost']
        mem_util = perfdata['mem_cost']
        bus_util = perfdata['sat_cost']
        
                    
        y_data = [cpu_util, gpu_util, bus_util]
        pos = np.arange(1, len(y_data)+1)
        xticklbls = [
                     'cpu_cost',
                     'gpu_cost',
                     #'mem_cost',
                     'bus_cost'
                     ]
        axs[ix].boxplot(y_data, positions=pos)
        axs[ix].set_xticks(pos)
        ymax =   80.0 if np.max([np.max(m) for m in y_data])<80.0 else np.max([np.max(m) for m in y_data])
                        
        axs[ix].set_ylim([-0.5, ymax])
        axs[ix].set_title(each_scenario)
    
       
    axs[-1].set_xticklabels(xticklbls, rotation=35)
    axs[-2].set_xticklabels(xticklbls, rotation=35)
    axs[-3].set_xticklabels(xticklbls, rotation=35)
    





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
#SCENARIO_ID = "launcher0" 
#DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
#MIF_FREQ = "default"
#INT_FREQ = "default"
#MIF_FREQ = 400000
#INT_FREQ = 50000

#cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
#mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)

#(count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)

#lbl = "{0}:mif-{1}:int-{2}".format(SCENARIO_ID, MIF_FREQ, INT_FREQ)


plot_freq_time_in_state(scenario_list, 'gpu_freq', TMP_MIF_FREQ='400000', TMP_INT_FREQ='400000')

   

plt.show()
print "-- Finished --"
