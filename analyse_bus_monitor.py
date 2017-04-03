import sys
import csv
import pprint
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"


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

SAMPLING_PERIOD = 50


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

def load_csv(fname):
    
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
            if MIF_FREQ=="default" and INT_FREQ=="default":
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
def plot_bus_data(perfdata, met_id, lbl, fname, units):    
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
                 marker=default_markers[each_metric], label=each_metric)
        plt.hold(True)
        total_data.append(filtered_data[each_metric])
    
    # show overall saturation as well 
    if met_id == MET_ID_SAT:
        sum_data_per_point = np.sum(total_data, axis=0)
        plt.plot(x_data, sum_data_per_point, color='k', 
                 marker='x', label='Total', lw=1.0)
    
    
    
        # testing
#         print "--"
#         for ix, each_metric in enumerate(default_bwmon_metric_order):
#             print each_metric, filtered_data[each_metric][100:105]
#         print sum_data_per_point[100:105]
#         print "--"
#         pprint.pprint(sum_data_per_point)  
    
    
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
    
    
# this is the frequency as set via devfreq
def plot_mem_freq_data(perfdata, lbl, fname, units):    
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



#################
#    MAIN code
#################

SCENARIO_ID = "llrand" 
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
MIF_FREQ = "default"
INT_FREQ = "default"
#MIF_FREQ = 100000
#INT_FREQ = 50000

csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)

(count, perfdata) = load_csv(csv_fname)

lbl = "{0}:mif-{1}:int-{2}".format(SCENARIO_ID, MIF_FREQ, INT_FREQ)
fname="plot_bus-{0}-".format(lbl)


plot_bus_data(perfdata, MET_ID_SAT, lbl, fname+"Saturation", "Bus saturation %")
#plot_bus_data(perfdata, MET_ID_BW, lbl, fname+"Bandwidth", "Bus bandwidth (MBps)")
#plot_bus_data(perfdata, MET_ID_FREQ,lbl, fname+"Bus Calculated Frequency", "Frequency (MHz)" )

if MIF_FREQ=="default" and INT_FREQ=="default":
    plot_mem_freq_data(perfdata, lbl, fname, "Frequency (MHz)")

plt.show()

print "-- Finished --"







