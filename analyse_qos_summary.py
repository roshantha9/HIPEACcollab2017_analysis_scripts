import sys
import csv
import pprint
import collections
import operator
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

#import seaborn.apionly as sns
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')

import matplotlib.patches as patches

BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

from common import DEFAULT_COL_LIST, scenario_list_fps, all_mifint_freqs_macroworkload
from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL
from analyse_fps_dumpsys import load_csv
from analyse_music_latency import load_log_music0
from analyse_ftp_latency import load_log_ftp0
from analyse_fps_dumpsys import load_csv as load_csv_ffmpeg0


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

def _normalise_list_sum(lst):
    norm = [float(i)/max(lst) for i in lst]
    return norm


####################
# Plotting related
####################
def plot_qos_comparison_allsc_allfr(all_fps_summary_data,
                                    all_music0_summary_data,
                                    all_ftp0_summary_data,
                                    all_ffmpeg0_summary_data, 
                                    sc_target_fps, scenario_order):
    
    fig, axs = plt.subplots(1,1, figsize=(8*1.2, 4.5*1.2))
    width = 0.5
    
    # fps related scenarios
    all_ydata = collections.OrderedDict()
    for ix, each_scenario in enumerate(sc_target_fps):
        all_ydata[each_scenario] = collections.OrderedDict()
        data_per_scenario = all_fps_summary_data[each_scenario]        
        for each_f, each_v in data_per_scenario.iteritems():
            if each_f in all_mifint_freqs_macroworkload:
                all_ydata[each_scenario][each_f] = np.mean(all_fps_summary_data[each_scenario][each_f])
            else:
                pass # ignore
            
    # music0
    all_ydata['music0'] = collections.OrderedDict()
    freqs = all_music0_summary_data.keys()
    for each_f in freqs:
        if each_f in all_mifint_freqs_macroworkload:
            all_ydata['music0'][each_f] =  np.mean(all_music0_summary_data[each_f]['tot_writes_diff'])
        else:
            pass # ignore
    
    # ftp0
    all_ydata['ftp0'] = collections.OrderedDict()
    freqs = all_ftp0_summary_data.keys()
    for each_f in freqs:
        if each_f in all_mifint_freqs_macroworkload:
            all_ydata['ftp0'][each_f] =  np.mean(all_ftp0_summary_data[each_f]['t_user_sys'])
        else:
            pass # ignore
    
    # ffmpeg0
    all_ydata['ffmpeg0'] = collections.OrderedDict()
    freqs = all_ffmpeg0_summary_data.keys()
    for each_f in freqs:
        if each_f in all_mifint_freqs_macroworkload:
            all_ydata['ffmpeg0'][each_f] =  np.mean(all_ffmpeg0_summary_data[each_f])
        else:
            pass # ignore
    
    
    pos = np.arange(0.7,(0.7*len(scenario_order))+0.7,0.7)
    print pos
    for ix, each_scenario in enumerate(scenario_order):
        print each_scenario
        sorted_all_ydata_tuples = sorted(all_ydata[each_scenario].items(), key=operator.itemgetter(1), reverse=True)
        ydata = [v[1] for v in sorted_all_ydata_tuples]
        
        ydata_norm = _normalise_list_sum(ydata)
        
        freqs = [v[0] for v in sorted_all_ydata_tuples]
        cols = [DEFAULT_COL_LIST[f] for f in freqs]
        
        x_data = [pos[ix]]*len(sorted_all_ydata_tuples)
        axs.bar(x_data, ydata_norm, width, color=cols)
    
    # legend
    rect_lbl_list = [s.replace("000", "") for s in all_mifint_freqs_macroworkload]
    cols = [DEFAULT_COL_LIST[f] for f in all_mifint_freqs_macroworkload]
    rects_list = []
    for ix, each_rect in enumerate(rect_lbl_list):
        rec = patches.Rectangle( (0.72, 0.1), 0.2, 0.6, facecolor=cols[ix])
        rects_list.append(rec)
    
    leg = plt.figlegend( rects_list, rect_lbl_list, loc = 'upper center', 
                         ncol=len(rects_list)/2, labelspacing=0. , fontsize=13,
                         frameon=False)
    leg.get_frame().set_facecolor('#FFFFFF')
    leg.get_frame().set_linewidth(0.0)
    leg.draggable()
    
    
    xticks = pos
    yticks = np.arange(0,1+0.1,0.1)
    axs.set_xticks(xticks + (width/2.))
    axs.set_xticklabels(scenario_order, fontsize=14, rotation=45)
    axs.set_xlim(0.7-(width/2.), (0.7*len(scenario_order))+0.7)
    
    
    axs.set_yticks(yticks)
    axs.set_yticklabels([str(y) for y in yticks], fontsize=14)
    axs.set_ylabel("Normalised QoS")
    
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    
    
    plt.subplots_adjust(top=0.90, left=0.07, right=.99, bottom=0.17, hspace=0.20, wspace=0.20)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   


    
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
SCENARIO_IDS = [
                # idle                 
                "launcher0",
                
                # multimedia                
                "ffmpeg0", "vlcplayer0", 
                "camera0", "camera1",
                "music0",
                
                # comms
                "line0", "line1", "line2",
                
                # social media (swiping)
                "facebook0", "facebook1",

                # browser (swiping)
                "chrome0",
                
                # downloading (background)
                "ftp0",
                
                # gaming
                "game0",                
            ]



######## fps related scenes ##########

METRIC = 'adjusted_fps'
# get and filter all scenario data for all freqs
cropped_data_list_fps = collections.OrderedDict() # cropped
sc_target_fps = {}
for each_scenario in scenario_list_fps:
    cropped_data_list_fps[each_scenario] = collections.OrderedDict()
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
        cropped_data_list_fps[each_scenario][each_freq_str] = cdata
        
    
########### music0 ##############
SCENARIO_ID = 'music0'
DEVFREQ_MIFINT_PAIRS = [k.split("-") for k in CUSTOM_CROPPING_PARAMS_ALL[SCENARIO_ID].keys()]
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"

music0_data = {} # raw    
freq_str_list = []
for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
    freq_str = '{0}-{1}'.format(mif_freq, int_freq)
    req_str = '{0}-{1}'.format(mif_freq, int_freq)
    freq_str_list.append(freq_str)        
    data_fname = DATA_DIR + "data_audio-{0}-{1}.txt".format(mif_freq, int_freq)
     
    music0_data[freq_str] = load_log_music0(data_fname)


########### ftp0 ##############
SCENARIO_ID = 'ftp0'
DEVFREQ_MIFINT_PAIRS = [k.split("-") for k in CUSTOM_CROPPING_PARAMS_ALL[SCENARIO_ID].keys()]
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"

ftp0_data = {} # raw    
freq_str_list = []
for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
    freq_str = '{0}-{1}'.format(mif_freq, int_freq)
    req_str = '{0}-{1}'.format(mif_freq, int_freq)
    freq_str_list.append(freq_str)        
    data_fname = DATA_DIR + "data_ftp-{0}-{1}.txt".format(mif_freq, int_freq)
     
    ftp0_data[freq_str] = load_log_ftp0(data_fname)

########### ffmpeg0 ##############
METRIC = 'ffmpeg_speed'
SCENARIO_ID = "ffmpeg0"
# get and filter all scenario data for all freqs
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
DEVFREQ_MIFINT_PAIRS = [k.split("-") for k in CUSTOM_CROPPING_PARAMS_ALL[SCENARIO_ID].keys()]
# gather data (csv)
data_list = [] # raw
cropped_data_list_ffmpeg0 = collections.OrderedDict()
freq_str_list = []
for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
    freq_str = '{0}-{1}'.format(mif_freq, int_freq)
    req_str = '{0}-{1}'.format(mif_freq, int_freq)
    freq_str_list.append(freq_str)        
    data_fname = DATA_DIR + "data_ffmpeg-{0}-{1}.log".format(mif_freq, int_freq)        
    # get csv data    
    (count, data) = load_csv_ffmpeg0(fname_fps=None, fname_ffmpeg=data_fname)
    # for fps
    data_list.append(data[METRIC])
    data_sample_size = len(data[METRIC])    
    
    cropped_data_list_ffmpeg0[freq_str] = data[METRIC]  
    



#sc_order = scenario_list_fps + ['music0'] + ['ftp0'] + ['ffmpeg0']

        
plot_qos_comparison_allsc_allfr(cropped_data_list_fps, music0_data, ftp0_data, cropped_data_list_ffmpeg0,
                                sc_target_fps, SCENARIO_IDS)


plt.show()

print "-- Finished --"







