import sys
import csv
import pprint
import collections
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

#import seaborn.apionly as sns
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

from common import DEFAULT_COL_LIST, scenario_list_fps
from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL
from analyse_fps_dumpsys import load_csv

SAMPLING_PERIOD = 1.6
NUM_CPU_CORES = 4

default_metric_order = [
                        "util",
                        "freq"
                        ]


default_colours = {
                   #"cpu_cores_util" : sns.color_palette("Blues", n_colors=NUM_CPU_CORES),
                   #"cpu_util" : sns.color_palette("Blues", n_colors=1),                   
                   #"gpu_util" : sns.color_palette("Reds", n_colors=1),
                   
                   "sf_fps" : "b",
                   "ffmpeg_fps" : "g"
                                                          
                   }






def _normalise_list(lst, norm_min=None, norm_max=None):    
    if norm_max == None:
        norm_max = np.max(lst)    
    if norm_min == None:
        norm_min = np.min(lst)    
    new_list = []
    for each_l in lst:   
        x = each_l     
        norm_val = (x-norm_min)/(norm_max-norm_min)
        new_list.append(norm_val)
    return new_list
    

####################
# Data manipulation
####################




####################
# Plotting related
####################
def plot_avg_metric(all_data, sc_target_fps, metric):
    cmap = plt.get_cmap('Reds')    
    #fig, axs = plt.subplots(int(np.ceil(len(all_data.keys())/3.0)),3, figsize=(10*1.2, 10*1.2))
    fig, axs = plt.subplots(int(np.ceil(len(all_data.keys())/6.0)),6)
    fig.canvas.set_window_title("plot_avg_metric - "+metric)    
    axs = axs.ravel()
    
    width = 0.4
    for ix, each_scenario in enumerate(scenario_list_fps):
        data_per_scenario = all_data[each_scenario]
        sorted_freq_list =  sorted(data_per_scenario.keys())
        colsd = [cmap(i) for i in np.linspace(0, 1, len(sorted_freq_list))]
        
        #y_data =   [ _non_zero_avg(all_data[each_scenario][f]) for f in sorted_freq_list]
        y_data =   [ np.mean(all_data[each_scenario][f]) for f in sorted_freq_list]
        y_data_norm = _normalise_list(y_data, norm_min=0.0, norm_max=sc_target_fps[each_scenario])
        #y_data_norm = y_data
        x_data = np.arange(1, len(sorted_freq_list)+1)
        print y_data_norm
        
        xticklbls=[]
        for f in sorted_freq_list:
            if "default" not in f:
                f = f.split("-")
                lbl = str(int(f[0])/1000) + "\n" + str(int(f[1])/1000)
            else:
                f = f.split("-")
                lbl = f[0] + "\n" + f[1]
                
            xticklbls.append(lbl)
            
        c = [cmap(0.7)]*(len(sorted_freq_list)-1) + [cmap(1.0)]
        rects1 = axs[ix].bar(x_data, y_data_norm, width, color=c)
        #bar_autolabel(rects1, axs[ix])
    
        axs[ix].set_xticks(x_data+(width/2.))
        axs[ix].set_xticklabels(xticklbls, fontsize=10)
        axs[ix].set_title(each_scenario, fontsize=12)
        axs[ix].set_xlim(width*1.5, len(x_data)+(width*2))
    
    
    # adjust odd num of subplots
    nsubplots = len(scenario_list_fps)     
    if (nsubplots % 2 != 0):
        for l in axs[-1].get_xaxis().get_majorticklabels():
            l.set_visible(True)
        fig.delaxes(axs[-1]) 
    
    plt.subplots_adjust(top=0.98, left=0.03, right=.99, bottom=0.03, hspace=0.37, wspace=0.18)
    
    
def _non_zero_avg(data):
    new_data = []
    for d in data:
        if d > 0:
            new_data.append(d)
        else:
            pass
    
    new_data_mean = np.mean(new_data)
    return new_data_mean
    


#################
#    MAIN code
#################
METRIC = 'adjusted_fps'
# get and filter all scenario data for all freqs
cropped_data_list = collections.OrderedDict() # cropped
sc_target_fps = {}
for each_scenario in scenario_list_fps:
    cropped_data_list[each_scenario] = collections.OrderedDict()
    DATA_DIR = BASE_DATA_DIR + each_scenario + "/"
    DEVFREQ_MIFINT_PAIRS = [k.split("-") for k in CUSTOM_CROPPING_PARAMS_ALL[each_scenario].keys()]
    # gather data (csv)
    data_list = [] # raw
    
    freq_str_list = []
    for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
        freq_str = '{0}-{1}'.format(mif_freq, int_freq)
        req_str = '{0}-{1}'.format(mif_freq, int_freq)
        freq_str_list.append(freq_str)        
        data_fname = DATA_DIR + "data_fps-{0}-{1}.csv".format(mif_freq, int_freq)        
        # get csv data    
        (count, data) = load_csv(fname_fps=data_fname, fname_ffmpeg=None)
        # for fps
        data_list.append(data[METRIC])
        data_sample_size = len(data[METRIC])    
        
        if each_scenario not in sc_target_fps:
            sc_target_fps[each_scenario] = data['target_fps']
        else:
            pass
        
    # cropping ends off
    for each_data, each_freq_str in zip(data_list, freq_str_list):    
        cdata = each_data[2:-3]
        cropped_data_list[each_scenario][each_freq_str] = cdata
        
    

        
plot_avg_metric(cropped_data_list, sc_target_fps, METRIC)


plt.show()

print "-- Finished --"







