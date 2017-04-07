import sys
import csv
import json
import pprint
import matplotlib
import collections
import operator
import numpy  as np
from numpy.dual import norm
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')

import matplotlib.patches as patches

from common import DEFAULT_COL_LIST, all_mifint_freqs_macroworkload
from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL



MOV_AVG_WIN_SIZE = 128

BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/MonsoonDataCapture/060317/"


LOAD_FILE = True
DATADUMP_FNAME = "summary_data/datadump_analyse_power_summary.json"

POW_SAMPLE_PERIOD = 0.0002


######################
#  Misc
######################
def load_csv(fname):
    time = []
    avg_pow = []
    with open(fname, 'rb') as f:
        reader = csv.reader(f)
        count=0
        for row in reader:
            if count > 0:
                time.append(float(row[0]))
                avg_pow.append(float(row[1]))
            
            count+=1
    return (time[-1], avg_pow)


def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def str_2_mifint(mifint_str):
    mif, int = mifint_str.split("-")
    return (mif,int)

def _write_formatted_file(fname, data, format="json"):        
        if(format == "pretty"):
            logfile=open(fname, 'w')
            pprint(data, logfile, width=128)
            
        elif(format == "json"):
            logfile=open(fname, 'w')
            json_data = json.dumps(data)
            logfile.write(json_data)
            
        else:
            logfile=open(fname, 'w')
            pprint(data, logfile, width=128)

def _read_json_file(fname):
    json_data=open(fname)
    pdata = json.load(json_data)
    return pdata

######################
#  Data manipulation
######################
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

def _normalise_list_sum(lst):
    norm = [float(i)/max(lst) for i in lst]
    return norm

def _normalise_list_against_val(lst, v):
    norm = [float(i)/v for i in lst]
    return norm


#################
#  Plotting
#################
def plot_pow_all_scenarios_all_freqs(all_summary_data, scenario_order):
    cmap = plt.get_cmap('Blues')
    
    
    fig, axs = plt.subplots(4,4, figsize=(10*1.2, 10*1.2))    
    axs = axs.ravel()
    
    width = 0.4
    for ix, each_scenario in enumerate(scenario_order):
        data_per_scenario = all_summary_data[each_scenario]
        sorted_freq_list =  sorted(data_per_scenario.keys())
        colsd = [cmap(i) for i in np.linspace(0, 1, len(sorted_freq_list))]
        
        y_data =   [all_summary_data[each_scenario][f]['mean'] for f in sorted_freq_list]
        y_data_norm = _normalise_list_sum(y_data)
        #y_data_norm = y_data
        x_data = np.arange(1, len(sorted_freq_list)+1)
        
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
       
    plt.subplots_adjust(top=0.98, left=0.03, right=.99, bottom=0.03, hspace=0.37, wspace=0.18)
        
        
def plot_pow_all_scenarios_all_freqs_stackedbar(all_summary_data, scenario_order):
    
    fig, axs = plt.subplots(1,1, figsize=(8*1.2, 4.5*1.2))
    width = 0.5
    
    # remove certain freqs, make simple key/val dict
    all_ydata = collections.OrderedDict()
    for ix, each_scenario in enumerate(scenario_order):
        all_ydata[each_scenario] = collections.OrderedDict()
        data_per_scenario = all_summary_data[each_scenario]        
        for each_f, each_v in data_per_scenario.iteritems():
            if each_f in all_mifint_freqs_macroworkload:
                all_ydata[each_scenario][each_f] = all_summary_data[each_scenario][each_f]['mean']
            else:
                pass # ignore
    
    pos = np.arange(0.7,(0.7*len(scenario_order))+0.7,0.7)
    print pos
    for ix, each_scenario in enumerate(scenario_order):
        print each_scenario
        sorted_all_ydata_tuples = sorted(all_ydata[each_scenario].items(), key=operator.itemgetter(1), reverse=True)
        ydata = [v[1] for v in sorted_all_ydata_tuples]
        
        #default_val = all_ydata[each_scenario]['default-default']        
        #ydata_norm = _normalise_list_against_val(ydata, default_val)
        
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
    axs.set_ylabel("Normalised power consumption")
    
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    
    
    plt.subplots_adjust(top=0.90, left=0.07, right=.99, bottom=0.17, hspace=0.20, wspace=0.20)


    
    
    


def plot_pow_imprv_comparison(all_pow_summary, scenario_order,
                              metric1,
                              metric2,
                              titlelbl,
                              ):
    fig, ax = plt.subplots(figsize=(12*1.2, 4.5*1.2))
          
    width = 0.3
    y_data = [all_pow_summary[k][metric1] for k in scenario_order]
    print y_data
    x_data = np.arange(1, len(all_pow_summary.keys())+1)    
    rects1 = ax.bar(x_data, y_data, width, color='#9ecae1')
    #bar_autolabel(rects1, ax)
              
    y_data = [all_pow_summary[k][metric2] for k in scenario_order]
    print "---"
    print y_data
    x_data = np.arange(1, len(all_pow_summary.keys())+1)
    rects2 = ax.bar(x_data + width, y_data, width, color='#084594')
    #bar_autolabel(rects2, ax)
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('% Improvement')    
    ax.set_title(titlelbl)    
    ax.set_xticks(x_data + width )
    ax.set_xticklabels(scenario_order, rotation=15)

    l=ax.legend((rects1[0], rects2[0]), ('Total power', 'Mean power'))
    l.draggable()
    
    plt.subplots_adjust(top=0.95, left=0.06, right=0.99, bottom=0.10)


def plot_pow_mean_default(all_pow_summary, scenario_order):
    fig, ax = plt.subplots(figsize=(9*1.2, 4*1.2))
    width = 0.45
    
    scenario_order_sorted = sorted(all_pow_summary, key=lambda k: all_pow_summary[k]['pow_mean_default'])
    
    y_data = [float(all_pow_summary[k]['pow_mean_default'])/1000.0 for k in scenario_order_sorted]
    #y_data_normalised = _normalise_list(y_data)
    y_data_normalised = y_data
    x_data = np.arange(1, len(all_pow_summary.keys())+1)
    rects1 = ax.bar(x_data + width, y_data_normalised, width, color='#084594')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Mean power consumption (W)', fontsize=11)    
    #ax.set_title("Normalised mean power consumption", fontsize=11)
    ax.set_xticks(x_data + width + (width/2.0)) 
    ax.set_xticklabels(scenario_order_sorted, rotation=45, fontsize=11)
    plt.xlim((1-width+0.4), x_data[-1] + width + (width/2.0) + (1-width+0.4))
        
    plt.subplots_adjust(top=0.95, left=0.06, right=0.99, bottom=0.10)

    

    
def bar_autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.1f' % height,
                ha='center', va='bottom', fontsize=10)
        
#################
#    Reporting
#################
def report_stats(data, freq, scenario = "Test", print_report=True):    
    s = {
             "total" : np.sum(data),
             "mean" : np.mean(data),
             "std": np.std(data),
             "max-min" : (np.max(data), np.min(data)),
             "samples": len(data),
             "pw_mt" : np.sum(data)*(len(data)*POW_SAMPLE_PERIOD), # power * time
             "pw_dt" : np.sum(data)/(len(data)*POW_SAMPLE_PERIOD), # power / time
        }  
    
    if print_report==True:  
        print "------------"
        print "{0}, {1}".format(scenario, freq)
        print "------------"     
        pprint.pprint(s) 
        print ""
        
    return s


def summarise_power_stats(all_data, scenario_id):    
    filt_data = all_data[scenario_id]    
    
    # get freq with min, max - total power
    pow_tot_min_v = np.min([v["total"] for k,v in filt_data.iteritems() if k != "default-default"])
    pow_tot_min_k = [k for k,v in filt_data.iteritems() if v['total'] == pow_tot_min_v][0]
    
    pow_tot_max_v = np.max([v["total"] for k,v in filt_data.iteritems() if k != "default-default"])
    pow_tot_max_k = [k for k,v in filt_data.iteritems() if v['total'] == pow_tot_max_v][0]
    
    pow_tot_800MHz_v = filt_data['800000-800000']["total"]
    pow_tot_800MHz_k = '800000-800000'
        
    # get freq with min, max - mean power
    pow_mean_min_v = np.min([v["mean"] for k,v in filt_data.iteritems() if k != "default-default"])
    pow_mean_min_k = [k for k,v in filt_data.iteritems() if v['mean'] == pow_mean_min_v][0]
    
    pow_mean_max_v = np.max([v["mean"] for k,v in filt_data.iteritems() if k != "default-default"])
    pow_mean_max_k = [k for k,v in filt_data.iteritems() if v['mean'] == pow_mean_max_v][0]
    
    pow_mean_800MHz_v = filt_data['800000-800000']["mean"]
    pow_mean_800MHz_k = '800000-800000'
    
    
    
    # default - total and mean power variance
    pow_tot_default = filt_data['default-default']['total']
    pow_mean_default = filt_data['default-default']['mean']
    pow_tot_min_default_var_percent = float(pow_tot_default - pow_tot_min_v)/float(pow_tot_default) * 100.0
    pow_mean_min_default_var_percent = float(pow_mean_default - pow_mean_min_v)/float(pow_mean_default) * 100.0
        
    # max - total and mean power variance
    pow_tot_max = filt_data[pow_tot_max_k]['total']
    pow_mean_max = filt_data[pow_mean_max_k]['mean']
    pow_tot_min_max_var_percent = float(pow_tot_max - pow_tot_min_v)/float(pow_tot_max) * 100.0
    pow_mean_min_max_var_percent = float(pow_mean_max - pow_mean_min_v)/float(pow_mean_max) * 100.0
    
    
    # 800MHz - total and mean power variance
    pow_tot_800MHz = pow_tot_800MHz_v
    pow_mean_800MHz = pow_mean_800MHz_v
    pow_tot_min_800MHz_var_percent = float(pow_tot_800MHz - pow_tot_min_v)/float(pow_tot_800MHz) * 100.0
    pow_mean_min_800MHz_var_percent = float(pow_mean_800MHz - pow_mean_min_v)/float(pow_mean_800MHz) * 100.0
        
    
    summary_data = {
                    # total power
                    'pow_tot_min_freq' : (pow_tot_min_k,pow_tot_min_v),
                    'pow_tot_min_default_var_percent' : pow_tot_min_default_var_percent,
                    'pow_tot_min_max_var_percent' : pow_tot_min_max_var_percent,
                    'pow_tot_min_800MHz_var_percent' : pow_tot_min_800MHz_var_percent,
                    'pow_tot_default' : pow_tot_default,
                    'pow_tot_max' : (pow_tot_max_k, pow_tot_max),
                    
                    # mean power
                    'pow_mean_min_freq' : (pow_mean_min_k,pow_mean_min_v),
                    'pow_mean_min_default_var_percent' : pow_mean_min_default_var_percent,
                    'pow_mean_min_max_var_percent' : pow_mean_min_max_var_percent,
                    'pow_mean_min_800MHz_var_percent' : pow_mean_min_800MHz_var_percent,
                    'pow_mean_default' : pow_mean_default,
                    'pow_mean_max' : (pow_mean_max_k, pow_mean_max),
                    
                    # power distribution
                    'pow_dist_default' : None,
                    'pow_measure_dur' : filt_data['default-default']['samples'] * POW_SAMPLE_PERIOD
                                        
                    }
    
    return summary_data
    
    
    

def adjust_sample_endings(all_sample_lengths):
    min_len = np.min(all_sample_lengths)
    
    print "s, new_s"
    for s in all_sample_lengths:        
        print s, s-min_len+1
        

#################
#    MAIN code
#################


# prerequisites
COL_LIST = DEFAULT_COL_LIST
SCENARIO_IDS = [
                # idle
                "idle0", "idle1", 
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

##### Collect and format data #####
all_summary_data = collections.OrderedDict()

# load from file
if LOAD_FILE == True:
    all_summary_data = _read_json_file(DATADUMP_FNAME)

else:
    for each_scenario_id in SCENARIO_IDS:
        
        print "----" , each_scenario_id, "----"
        
        DATA_DIR = BASE_DATA_DIR + each_scenario_id + "/"
        CUSTOM_CROPPING_PARAMS = CUSTOM_CROPPING_PARAMS_ALL[each_scenario_id]
        DEVFREQ_MIFINT_PAIRS = [str_2_mifint(k) for k in CUSTOM_CROPPING_PARAMS.keys()]
        
        all_summary_data[each_scenario_id] = collections.OrderedDict()
        
        # gather data (csv)
        data_list = [] # raw
        cropped_data_list = [] # cropped
        freq_str_list = []
    
        all_min_sample_size = sys.maxint
    
        for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
            freq_str = '{0}-{1}'.format(mif_freq, int_freq)
            print freq_str
            freq_str_list.append(freq_str)
            data_fname = DATA_DIR+'pmdata_mif{0}_int{1}.csv'.format(mif_freq, int_freq)
            
            # get csv data
            (final_time, data) = load_csv(data_fname)
            data_list.append(data)
            data_sample_size = len(data)
            
            # used for cropping
            if (data_sample_size < all_min_sample_size):
                all_min_sample_size = data_sample_size
            else:
                pass
            
        # start cropping
        for each_data, each_freq_str in zip(data_list, freq_str_list):    
            cdata = each_data[CUSTOM_CROPPING_PARAMS[each_freq_str][0]:CUSTOM_CROPPING_PARAMS[each_freq_str][1]]
            cropped_data_list.append(cdata)
    
        # report data    
        for (each_data, each_freq) in zip(cropped_data_list, freq_str_list):
            all_summary_data[each_scenario_id][each_freq] = {}
            all_summary_data[each_scenario_id][each_freq] = report_stats(each_data, each_freq, 
                                                                        scenario=each_scenario_id,
                                                                        print_report=False)
                                            
    # save the data to dump file
    _write_formatted_file(DATADUMP_FNAME, all_summary_data)                                


##### Perform summary reporting #####
#pprint.pprint(all_summary_data)
all_pow_stat_summary = {}
for each_scenario_id in SCENARIO_IDS:
    #print "--- ", each_scenario_id, " ---"
    all_pow_stat_summary[each_scenario_id] = summarise_power_stats(all_summary_data, each_scenario_id) 
    #pprint.pprint( all_pow_stat_summary )
    #print ""
    

##### Plotting #####
# plot_pow_imprv_comparison(all_pow_stat_summary, SCENARIO_IDS,
#                               "pow_tot_min_default_var_percent",
#                               "pow_mean_min_default_var_percent",
#                               'Improvement over default memory governor using fixed MIF/INT frequencies',
#                               )
#  
# plot_pow_imprv_comparison(all_pow_stat_summary, SCENARIO_IDS,
#                               "pow_tot_min_800MHz_var_percent",
#                               "pow_mean_min_800MHz_var_percent",
#                               'Improvement over highest MIF/INT freq setting (800MHz,800MHz) using lower fixed MIF/INT frequencies',
#                               )
# 


#plot_pow_mean_default(all_pow_stat_summary, SCENARIO_IDS)


#plot_pow_all_scenarios_all_freqs(all_summary_data, SCENARIO_IDS)

plot_pow_all_scenarios_all_freqs_stackedbar(all_summary_data, SCENARIO_IDS)

plt.show()
        