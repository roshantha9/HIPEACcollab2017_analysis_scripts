import sys
import csv
import pprint
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

#import seaborn.apionly as sns
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"



SAMPLING_PERIOD = 200
NUM_CPU_CORES = 4

default_metric_order = [
                        "util",
                        "freq"
                        ]


default_colours = {
                   #"cpu_cores_util" : sns.color_palette("Blues", n_colors=NUM_CPU_CORES),
                   #"cpu_util" : sns.color_palette("Blues", n_colors=1),                   
                   #"gpu_util" : sns.color_palette("Reds", n_colors=1),
                   
                   "cpu_cores_util" : matplotlib.cm.get_cmap('Blues')(np.linspace(0,1,NUM_CPU_CORES+2))[2:],
                   "cpu_util" : matplotlib.cm.get_cmap('Blues')(np.linspace(0,1,2)),                   
                   "gpu_util" : 'Red',
                   
                   "cpu_freq" : "blue",
                   "gpu_freq" : "red"                                       
                   }

default_markers = {                                      
                   }
                     

MET_COREBUSY_ID         = 0
MET_CORENICEBUSY_ID     = 1
MET_COREIDLE_ID         = 2



def _check_row_extraction(data):    
    pass

def load_csv(fname):
    
    data = {            
            "cpu_freq" : [],
            "cpu_util" : [], # total cpu util
            "cpu_util_freq" : [], # total cpu util (taking freq into account)
            "cpu_util_freq_norm" : [], # total cpu util (taking freq into account) - normalised
            "cpu_core_stats" : {cid: {
                                      "busy":[],
                                      "nicebusy":[],
                                      "idle":[],
                                      "util":[], 
                                      "util_freq":[],
                                      "util_freq_norm" : [], # normalised 'util_freq'
                                      } for cid in xrange(NUM_CPU_CORES)},            
            "gpu_freq" : [],
            "gpu_util" : [],
            }    
    
    with open(fname, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
                
        for row in reader:
            sampleCount = row[0]
            #count, freq, {busy, nicebusy, idle} - per CPU, gpufreq, gpuutil
            
            # bw, freq, sat, duration                    
            data["cpu_freq"].append(float(row[1]))
            
            # @todo: fix this to work with n cores
#             for cid in xrange(NUM_CPU_CORES):
#                 print cid
#                 data["cpu_core_stats"][cid]["busy"].append(float(row[((cid*0) + MET_COREBUSY_ID) + 2]))
#                 data["cpu_core_stats"][cid]["nicebusy"].append(float(row[((cid*1) + MET_CORENICEBUSY_ID) + 2]))
#                 data["cpu_core_stats"][cid]["idle"].append(float(row[((cid*2) + MET_COREIDLE_ID) + 2]))
            
            cid=0
            data["cpu_core_stats"][cid]["busy"].append(float(row[2]))
            data["cpu_core_stats"][cid]["nicebusy"].append(float(row[3]))
            data["cpu_core_stats"][cid]["idle"].append(float(row[4]))
            
            cid=1
            data["cpu_core_stats"][cid]["busy"].append(float(row[5]))
            data["cpu_core_stats"][cid]["nicebusy"].append(float(row[6]))
            data["cpu_core_stats"][cid]["idle"].append(float(row[7]))
            
            cid=2
            data["cpu_core_stats"][cid]["busy"].append(float(row[8]))
            data["cpu_core_stats"][cid]["nicebusy"].append(float(row[9]))
            data["cpu_core_stats"][cid]["idle"].append(float(row[10]))
                
            cid=3
            data["cpu_core_stats"][cid]["busy"].append(float(row[11]))
            data["cpu_core_stats"][cid]["nicebusy"].append(float(row[12]))
            data["cpu_core_stats"][cid]["idle"].append(float(row[13]))
                
            
            data['gpu_freq'].append(row[14]) # in MHz
            data['gpu_util'].append((float(row[15])/256.0)*100.0)
                        
            #_check_row_extraction(data)
            
    return (sampleCount, data)


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
def calc_and_update_cpu_util(data):
    
    # calculate core specific util
    for cid in xrange(NUM_CPU_CORES):
        
        zeropad = np.array([0.0])
        
        # calculate core specific util
        cbusy = np.array(data["cpu_core_stats"][cid]["busy"])
        cnicebusy = np.array(data["cpu_core_stats"][cid]["nicebusy"])
        cidle = np.array(data["cpu_core_stats"][cid]["idle"])
        
        
        cbusysub = np.diff(cbusy)
        cbusysub = np.concatenate([zeropad, cbusysub])
        
        cnicebusysub = np.diff(cnicebusy)
        cnicebusysub = np.concatenate([zeropad, cnicebusysub])
        
        cidlesub = np.diff(cidle)
        cidlesub = np.concatenate([zeropad, cidlesub])
        
        cidlesub_plus_busysub = cbusysub + cidlesub
        
        ratio = np.nan_to_num((cbusysub / cidlesub_plus_busysub))
        #max_jiffy = np.array([21.0]*len(cbusysub))
        #ratio = np.nan_to_num((cbusysub / max_jiffy))
        
        cutil = ratio * 100.0
        
        #pprint.pprint(cbusy)
        #pprint.pprint(max_jiffy)
        #pprint.pprint(cutil)
#         print "========================="
        #x = raw_input("press..")
        
        #sys.exit()
        
        cutil_freq = ratio * np.array(data["cpu_freq"])
        
        #pprint.pprint(cutil_freq)
                
        data["cpu_core_stats"][cid]["util"] = cutil
        data["cpu_core_stats"][cid]["util_freq"] = cutil_freq
    
    #sys.exit()
    
    # calculate normlised CPU core util
    # find global min/max     
    all_min = 0
    all_max = 0
    for cid in xrange(NUM_CPU_CORES):
        cmax = np.max(data["cpu_core_stats"][cid]["util_freq"])
        if cmax > all_max:
            all_max = cmax
        else:
            pass
    
    
    for cid in xrange(NUM_CPU_CORES):        
        normed_util_freq = _normalise_list(data["cpu_core_stats"][cid]["util_freq"], 
                                           norm_min=all_min, norm_max=all_max)
        data["cpu_core_stats"][cid]["util_freq_norm"] = np.array(normed_util_freq)*100.0
    
        #pprint.pprint(normed_util_freq)
        print "------"
       
    # calculate total (overall) util
    overall_util = np.array([0.0]*len(data["cpu_core_stats"][0]["util"]))
    overall_util_freq = np.array([0.0]*len(data["cpu_core_stats"][0]["util_freq_norm"]))
    for cid in xrange(NUM_CPU_CORES):        
        overall_util += data["cpu_core_stats"][cid]["util"]
        overall_util_freq += data["cpu_core_stats"][cid]["util_freq_norm"]
    
    data["cpu_util"] = overall_util/float(NUM_CPU_CORES)
    data["cpu_util_freq"] = overall_util_freq/float(NUM_CPU_CORES)
    
    return data



####################
# Plotting related
####################
PLOT_UTIL_PER_CORE = False
def plot_util_data(perfdata, fname, lbl):    
    fig = plt.figure(figsize=(8*1.2, 4*1.2))
    fig.canvas.set_window_title(fname)
    
    cpu_util = perfdata["cpu_util"]
    cpu_core_util = {cid: perfdata["cpu_core_stats"][cid]["util_freq_norm"] for cid in xrange(NUM_CPU_CORES)}    
    gpu_util = perfdata["gpu_util"]
    
    samples = len(perfdata["cpu_util"])
    
    total_duration = (samples * SAMPLING_PERIOD)/1000.0
    
    x_data = np.linspace(0, total_duration, samples)
    
    # plot util per core
    if PLOT_UTIL_PER_CORE == True:
        for cid in xrange(NUM_CPU_CORES):
            #pprint.pprint(perfdata["cpu_core_stats"][cid]["util"])
            plt.plot(x_data, perfdata["cpu_core_stats"][cid]["util_freq_norm"], 
                     color=default_colours["cpu_cores_util"][cid], label="CPU-Core"+str(cid))
            plt.hold(True)
            
    # plot overall cpu util
    if PLOT_UTIL_PER_CORE == False:    
        plt.plot(x_data, perfdata["cpu_util"], 
                     color=default_colours["cpu_cores_util"][-1], label="CPU")
        plt.axhline(y=np.mean(perfdata["cpu_util"]), color=default_colours["cpu_cores_util"][-1], linestyle='--', 
                    alpha=0.5, linewidth=2.0)
        
    # plot gpu util    
    plt.plot(x_data, perfdata["gpu_util"], 
                 color=default_colours["gpu_util"], label="GPU")
    plt.axhline(y=np.mean(perfdata["gpu_util"]), color=default_colours["gpu_util"], linestyle='--', 
                    alpha=0.5, linewidth=2.0)
        
        
    l = plt.legend(ncol=6, fontsize=11, 
                   labelspacing=0, handletextpad=0.2, loc='upper center',
                   bbox_to_anchor=(0.5, 1.17))
    l.draggable()
    l.get_frame().set_facecolor('#FFFFFF')
    
    plt.title("CPU/GPU Utilisation")
    plt.xlabel("time (s)")
    plt.ylabel("Utilisation %")
    plt.ylim([-0.75, 100.75])
    
    font = {'family': 'serif',        
        'weight': 'normal',
        'size': 11,
        }
    
    plt.title(lbl, fontdict=font)
    
    plt.subplots_adjust(top=0.87, left=0.07, right=0.98, bottom=0.112)



def plot_freq_data(perfdata, fname, lbl):    
    fig = plt.figure(figsize=(8*1.2, 4*1.2))
    fig.canvas.set_window_title(fname)
    
    cpu_freq = [float(x)/1000.0 for x in perfdata["cpu_freq"]] # MHz        
    gpu_freq = [float(x) for x in perfdata["gpu_freq"]]
    
    samples = len(cpu_freq)    
    total_duration = (samples * SAMPLING_PERIOD)/1000.0    
    x_data = np.linspace(0, total_duration, samples)
    
    # plot freqs
    plt.plot(x_data, cpu_freq, color=default_colours["cpu_freq"], label="CPU Freq")    
    plt.hold(True)
    plt.plot(x_data, gpu_freq, color=default_colours["gpu_freq"], label="GPU Freq")
    plt.hold(True)
    
    # plot means    
    plt.axhline(y=np.mean(cpu_freq), color=default_colours["cpu_freq"], linestyle='--', alpha=0.5, lw=2.0)
    plt.hold(True)    
    plt.axhline(y=np.mean(gpu_freq), color=default_colours["gpu_freq"], linestyle='--', alpha=0.5, lw=2.0)
    
    # legend
    l = plt.legend(ncol=6, fontsize=11, 
                   labelspacing=0, handletextpad=0.2, loc='upper center',
                   bbox_to_anchor=(0.5, 1.17))
    l.draggable()
    l.get_frame().set_facecolor('#FFFFFF')
    
    plt.title("CPU/GPU Frequency")
    plt.xlabel("time (s)")
    plt.ylabel("Frequency (MHz)")
    
    font = {'family': 'serif',        
        'weight': 'normal',
        'size': 11,
        }
    
    plt.title(lbl, fontdict=font)
    
    plt.subplots_adjust(top=0.87, left=0.07, right=0.98, bottom=0.112)

#################
#    MAIN code
#################

SCENARIO_ID = "game0" 
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
MIF_FREQ = "default"
INT_FREQ = "default"
#MIF_FREQ = 100000
#INT_FREQ = 50000

csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)


(count, perfdata) = load_csv(csv_fname)
perfdata = calc_and_update_cpu_util(perfdata)

lbl = "{0}:mif-{1}:int-{2}".format(SCENARIO_ID, MIF_FREQ, INT_FREQ)
fname="plot_cpugpu-{0}-".format(lbl)

plot_util_data(perfdata, fname, lbl)
plot_freq_data(perfdata, fname, lbl)

plt.show()

print "-- Finished --"







