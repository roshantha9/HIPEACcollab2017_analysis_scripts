import sys
import csv
import pprint
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')


from common import scenario_list, roylongbottom_microbench_list, mif_int_freqstr_to_tuple

                    
from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL

BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

FIG_OUTPUT_DIR_BUSSTATS = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/plots/DetailedPlots/bus_stats/"
FIG_OUTPUT_DIR_BUSFREQ = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/plots/DetailedPlots/bus_freq/"


default_bwmon_metric_order = ["mfc0", "mfc1", 
                        "isp0", "isp1",
                        "gen", "fsys",
                        "mem0_0", "mem1_0",
                        "mem0_1", "mem1_1",           
                        "disp1"]

default_freq_metric_order = ["mif_freq", "int_freq"]

default_colours = {
                   "mfc0" : '#4daf4a', "mfc1" : '#4daf4a',                    
                   "isp0" : '#ff7f00', "isp1" : '#ff7f00',
                   "gen" : '#ffff33',
                   "fsys" : '#a65628',
                   "mem0_0" : '#377eb8', "mem1_0" : '#377eb8',
                   "mem0_1" : '#e41a1c', "mem1_1" : '#e41a1c',            
                   "disp1" : '#984ea3',
                   
                   # frequencies
                   "mif_freq" : "blue",
                   "int_freq" : "red"
                   }

default_markers = {
                   "mfc0" : '.', "mfc1" : 'x',                    
                   "isp0" : '.', "isp1" : 'x',
                   "gen" : '.',
                   "fsys" : '.',
                   "mem0_0" : '.', "mem1_0" : 'x',
                   "mem0_1" : '.', "mem1_1" : 'x',            
                   "disp1" : '.',       
                   
                   # frequencies
                   "mif_freq" : "x",
                   "int_freq" : "x"
                   }
                     

MET_ID_BW    = 0
MET_ID_FREQ  = 1
MET_ID_SAT   = 2
MET_ID_DUR   = 3

SAMPLING_PERIOD = 200


def _check_row_extraction(data):
    for k,v in data.iteritems():
        if (k != "mif_freq") and (k != "int_freq"): 
            rlen = len(v[-1])        
            if rlen != 4: 
                sys.exit("Error - incorrect row extraction")
            else:
                pass
        else:
            pass

def load_csv(fname, miffreq, intfreq):
    
    data = {            
            "mfc0" : [],
            "mfc1" : [],                    
            "isp0" : [],
            "isp1" : [],
            "gen" : [],
            "fsys" : [],
            "mem0_0" : [],
            "mem1_0" : [],
            "mem0_1" : [],
            "mem1_1" : [],            
            "disp1" : [],
            
            # frequencies
            "mif_freq" : [],
            "int_freq" : []        
            }    
    
    with open(fname, 'rb') as f:
        reader = csv.reader(f, delimiter=',')        
        for row in reader:
            sampleCount = row[0]
            #print sampleCount    
            
            # bw, freq, sat, duration                    
            data["mfc0"].append(row[1:5])
            data["mfc1"].append(row[5:9])
            data["isp0"].append(row[9:13])
            data["isp1"].append(row[13:17])
            data["gen"].append(row[17:21])
            data["fsys"].append(row[21:25])
            data["mem0_0"].append(row[25:29])
            data["mem1_0"].append(row[29:33])
            data["mem0_1"].append(row[33:37])
            data["mem1_1"].append(row[37:41])
            data["disp1"].append(row[41:45])
            
            # mif, int freqs
            if miffreq=="default" and intfreq=="default":                              
                data["mif_freq"].append(int(row[45]))
                data["int_freq"].append(int(row[46]))
                        
            _check_row_extraction(data)
            
    return (sampleCount, data)


####################
# Data manipulation
####################
def get_mem_data_by_metric(metric_id):
    filtered_data = {
                     "mfc0" : [float(v[metric_id]) for v in perfdata["mfc0"]],
                     "mfc1" : [float(v[metric_id]) for v in perfdata["mfc1"]],                    
                     "isp0" : [float(v[metric_id]) for v in perfdata["isp0"]],
                     "isp1" : [float(v[metric_id]) for v in perfdata["isp1"]],
                     "gen" : [float(v[metric_id]) for v in perfdata["gen"]],
                     "fsys" : [float(v[metric_id]) for v in perfdata["fsys"]],
                     "mem0_0" : [float(v[metric_id]) for v in perfdata["mem0_0"]],
                     "mem1_0" : [float(v[metric_id]) for v in perfdata["mem1_0"]],
                     "mem0_1" : [float(v[metric_id]) for v in perfdata["mem0_1"]],
                     "mem1_1" : [float(v[metric_id]) for v in perfdata["mem1_1"]],            
                     "disp1" : [float(v[metric_id]) for v in perfdata["disp1"]],
                     }
    
    return filtered_data


####################
# Plotting related
####################
def plot_bus_data(perfdata, met_id, lbl, fname, units, save_fig=False):    
    print fname
    fig = plt.figure(figsize=(6*1.2, 3.5*1.2))
    ax = fig.add_subplot(111)
    fig.canvas.set_window_title(fname)
    
    filtered_data = get_mem_data_by_metric(met_id)
    samples = len(filtered_data[filtered_data.keys()[0]])
    total_duration = (samples * SAMPLING_PERIOD)/1000.0
    
    x_data = np.linspace(0, total_duration, samples)
    
    all_max = 0
    total_data = []
    for ix, each_metric in enumerate(default_bwmon_metric_order):
        if (np.max(filtered_data[each_metric] > all_max)):
            all_max = np.max(filtered_data[each_metric])
            
        plt.plot(x_data, filtered_data[each_metric], color=default_colours[each_metric], 
                 marker=default_markers[each_metric], label=_rename_lbl(each_metric))
        plt.hold(True)
        total_data.append(filtered_data[each_metric])
    
    # show overall saturation as well 
    if met_id == MET_ID_SAT:
        sum_data_per_point = np.sum(total_data, axis=0)
        plt.plot(x_data, sum_data_per_point, color='k', 
                 marker='x', label='Total', lw=1.0)

    l = plt.legend(ncol=6, fontsize=11, frameon=False,
                   labelspacing=0, handletextpad=0.2, loc='upper center',
                   bbox_to_anchor=(0.5, 1.24))
    l.draggable()
    l.get_frame().set_facecolor('#FFFFFF')
    
    #plt.title(lbl)
    plt.xlabel("time (s)", fontsize=12)
    plt.ylabel(units, fontsize=12)
    plt.xlim(0.0, x_data[-1]+0.05)
    
    font = {      
        'weight': 'normal',
        'size': 12,
        }
    
    plt.title(lbl, fontdict=font)
    
    if(len(str(int(all_max)))>3): # xtick too large, pushes label out
        plt.subplots_adjust(top=0.85, left=0.115, right=0.98, bottom=0.13)
    else:
        plt.subplots_adjust(top=0.85, left=0.09, right=0.98, bottom=0.13)
    
    
    if save_fig == True:
        fig.savefig(FIG_OUTPUT_DIR_BUSSTATS + fname + ".png")
        fig.savefig(FIG_OUTPUT_DIR_BUSSTATS + fname + ".pdf")
    
# this is the frequency as set via devfreq
def plot_mem_freq_data(perfdata, lbl, fname, units, save_fig=False):    
    print fname
    fig = plt.figure(figsize=(6*1.2, 3.5*1.2))
    ax = fig.add_subplot(111)
    fig.canvas.set_window_title(fname)
    
    # convert KHz to MHz 
    filtered_data = {
                "mif_freq": [float(x)/1000.0 for x in perfdata["mif_freq"]],
                "int_freq": [float(x)/1000.0 for x in perfdata["int_freq"]],
                }
    
    samples = len(filtered_data[filtered_data.keys()[0]])
    total_duration = (samples * SAMPLING_PERIOD)/1000.0
    
    x_data = np.linspace(0, total_duration, samples)
    
    all_max = 0
    
    for ix, each_metric in enumerate(default_freq_metric_order):
        if (np.max(filtered_data[each_metric] > all_max)):
            all_max = np.max(filtered_data[each_metric])
        
        plt.plot(x_data, filtered_data[each_metric], color=default_colours[each_metric], 
                 marker=default_markers[each_metric], label=each_metric)
        plt.hold(True)
    
    l = plt.legend(ncol=6, fontsize=11, 
                   labelspacing=0, handletextpad=0.2, loc='upper center',
                   bbox_to_anchor=(0.5, 1.24))
    l.draggable()
    l.get_frame().set_facecolor('#FFFFFF')
    
    #plt.title(lbl)
    plt.xlabel("time (s)")
    plt.ylabel(units)
    plt.xlim(0.0, x_data[-1]+0.05)
    
    font = {'family': 'serif',        
        'weight': 'normal',
        'size': 11,
        }
    
    plt.title(lbl, fontdict=font)
    
    if(len(str(int(all_max)))>3): # xtick too large, pushes label out
        plt.subplots_adjust(top=0.85, left=0.115, right=0.98, bottom=0.13)
    else:
        plt.subplots_adjust(top=0.85, left=0.09, right=0.98, bottom=0.13)
    
    if save_fig == True:
        fig.savefig(FIG_OUTPUT_DIR_BUSFREQ + fname + ".png")
        fig.savefig(FIG_OUTPUT_DIR_BUSFREQ + fname + ".pdf")

 
def _rename_lbl(m):     
    if m == "mem0_0" : result = "gfx-mem0"
    elif m == "mem1_0" : result = "gfx-mem1"    
    elif m == "mem0_1" : result = "cpu-mem0"
    elif m == "mem1_1" : result = "cpu-mem1"    
    else : result = m
    
    return result



#################
#    MAIN code
#################

all_sc_list = scenario_list + roylongbottom_microbench_list
all_freq_list = CUSTOM_CROPPING_PARAMS_ALL

for each_sc in all_sc_list:
    
    SCENARIO_ID = each_sc 
    DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
    
    for each_mifint_freq in all_freq_list[each_sc]:
        
        [MIF_FREQ, INT_FREQ] = mif_int_freqstr_to_tuple(each_mifint_freq)
        
        lbl = "{0}--MIF-{1}:INT-{2}".format(SCENARIO_ID, MIF_FREQ, INT_FREQ)
        fname="plot_bus-{0}-".format(lbl)
        
        csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
        
        (count, perfdata) = load_csv(csv_fname, MIF_FREQ, INT_FREQ)
        
        plot_bus_data(perfdata, MET_ID_SAT, lbl, fname+"Saturation", "Bus saturation %", save_fig=True)
        plot_bus_data(perfdata, MET_ID_BW, lbl, fname+"Bandwidth", "Bus bandwidth (MBps)", save_fig=True)
        plot_bus_data(perfdata, MET_ID_FREQ,lbl, fname+"BusCalcFreq", "Frequency (MHz)" , save_fig=True)
        
        if MIF_FREQ=="default" and INT_FREQ=="default":
            plot_mem_freq_data(perfdata, lbl, fname+"MIFINTFreq", "Frequency (MHz)", save_fig=True)
        elif ("test" in MIF_FREQ) and ("test" in INT_FREQ):
            plot_mem_freq_data(perfdata, lbl, fname+"MIFINTFreq", "Frequency (MHz)", save_fig=True)
        else:
            pass
        
        #sys.exit()

#plt.show()

print "-- Finished --"







