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
                     roylongbottom_microbench_list,\
                     rlbench_test_mapping, roylongbottom_microbench_list_renames,\
                     CPU_FREQS_KHZ, GPU_FREQS_MHZ, INT_FREQS_KHZ, MIF_FREQS_KHZ


from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL

BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

NUM_CPU_CORES = 4
SAMPLING_PERIOD = 200


    
    
####################
# Data manipulation
####################
def _normalise_list_sum(lst):
    norm = [float(i)/max(lst) for i in lst]
    return norm

def get_allfreq_list(f_metric):
    cmap = plt.get_cmap('autumn')
    freq_list = []
    if f_metric == 'cpu_freq' :        
        colsd = {f: cmap(i) for (f, i) in zip(CPU_FREQS_KHZ, np.linspace(0, 1, len(CPU_FREQS_KHZ)))}        
        freq_list = CPU_FREQS_KHZ
    elif f_metric == 'gpu_freq' :        
        colsd = {f: cmap(i) for (f, i) in zip(GPU_FREQS_MHZ, np.linspace(0, 1, len(GPU_FREQS_MHZ)))}
        freq_list = GPU_FREQS_MHZ
    elif f_metric == 'bus_mif_freq' :        
        colsd = {f: cmap(i) for (f, i) in zip(MIF_FREQS_KHZ, np.linspace(0, 1, len(MIF_FREQS_KHZ)))}        
        freq_list = MIF_FREQS_KHZ        
    elif f_metric == 'bus_int_freq' :
        colsd = {f: cmap(i) for (f, i) in zip(INT_FREQS_KHZ, np.linspace(0, 1, len(INT_FREQS_KHZ)))}
        freq_list = INT_FREQS_KHZ          
    else:
        sys.exit("Error - invalid freq type")    
        
    
    return freq_list, colsd




def get_freqstats_per_scenario_per_test(sc_list, freq_metrics):
    print "get_freqstats_per_scenario_per_test:Enter"
    
    all_freqdist_data = collections.OrderedDict()
    for ix, each_scenario in enumerate(sc_list):
        print each_scenario       
        all_freqdist_data[each_scenario] = collections.OrderedDict() 
        for each_mifint_freq_str in CUSTOM_CROPPING_PARAMS_ALL[each_scenario].keys():
            all_freqdist_data[each_scenario][each_mifint_freq_str] = {} 
            
            DEVFREQ_MIFINT_PAIRS = [int(s) if (("default" not in s) and ("test" not in s)) else s for s in each_mifint_freq_str.split("-")]
            MIF_FREQ = DEVFREQ_MIFINT_PAIRS[0]
            INT_FREQ = DEVFREQ_MIFINT_PAIRS[1]
            
            DATA_DIR = BASE_DATA_DIR + each_scenario + "/"        
            cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
            mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
            
            (c, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)
            
            for each_fmetric in freq_metrics:
                
                freq_data_dict = {
                     'sum' : np.sum(perfdata[each_fmetric]),
                     'mode' : _counter_mode(perfdata[each_fmetric]),
                     'mean' : np.mean(perfdata[each_fmetric]),
                     'min' : np.min(perfdata[each_fmetric]),
                     'max' : np.max(perfdata[each_fmetric]),                     
                     'counter' : collections.Counter(perfdata[each_fmetric]),
                     'transitions' : np.count_nonzero(np.diff(perfdata[each_fmetric]))
                     }                    
                all_freqdist_data[each_scenario][each_mifint_freq_str][each_fmetric] = freq_data_dict

    return all_freqdist_data



# positive/negative slopes
def get_metric_slope_analysis(sc_list):
    print "get_freqstats_per_scenario_per_test:Enter"
    
    all_freqdist_data = {}
    for ix, each_scenario in enumerate(sc_list):
        print each_scenario       
        all_freqdist_data[each_scenario] = collections.OrderedDict() 
        
        DEVFREQ_MIFINT_PAIRS = ["default", "default"]
        MIF_FREQ = DEVFREQ_MIFINT_PAIRS[0]
        INT_FREQ = DEVFREQ_MIFINT_PAIRS[1]
        
        DATA_DIR = BASE_DATA_DIR + each_scenario + "/"        
        cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
        mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
        
        (c, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)
        
        interested_metrics = ['cpu_util', 'sat_total', 'bus_mif_freq', 'bus_int_freq', 'cpu_freq']
        
        # get slope for interested metrics
        tmp_dict = {} # 1 or -1 or 0 per tick
        for each_metric in interested_metrics:
            tmp_dict[each_metric] = []
            diff = np.diff(perfdata[each_metric])
            
            # decrease precision
            if (each_metric == "cpu_util") or (each_metric == "sat_total"):
                diff = [x if abs(x) > 5.0 else 0.0  for x in diff]
            else:
                pass
            
            
            for x in diff:
                if x==0: # no change
                    tmp_dict[each_metric].append(0)                    
                elif x>0: # positive        
                    tmp_dict[each_metric].append(1)                        
                elif x<0: # negative
                    tmp_dict[each_metric].append(-1)
                else:
                    pass
        
        # construct result [1 2 3 4]       
        samples = len(tmp_dict[interested_metrics[0]])
        result = []
        for i in xrange(samples):
            if (tmp_dict['cpu_util'][i] == 0) and (tmp_dict['sat_total'][i]==-1) and (tmp_dict['bus_mif_freq'][i]==-1) and (tmp_dict['cpu_freq'][i]==1):
                result.append(1)
            elif (tmp_dict['cpu_util'][i] == -1) and (tmp_dict['sat_total'][i]==-1) and (tmp_dict['bus_mif_freq'][i]==-1) and (tmp_dict['cpu_freq'][i]==0):
                result.append(2)
            
            elif (tmp_dict['cpu_util'][i] in [0,-1]) and (tmp_dict['bus_mif_freq'][i]==-1) and (tmp_dict['cpu_freq'][i]==0):
                result.append(3)
            
            elif (tmp_dict['cpu_util'][i] in [0,-1]) and (tmp_dict['bus_mif_freq'][i]==-1) and (tmp_dict['cpu_freq'][i]==1):
                result.append(4)
            
            else:
                result.append(0)
        
        
        all_freqdist_data[each_scenario] = {
#                                             'trans_type1' : result.count(1),
#                                             'trans_type2' : result.count(2),
#                                             'trans_type3' : result.count(2),
#                                             'trans_type4' : result.count(2),
                                            
                                            'trans_type1_perc' : float(result.count(1))/float(len(result)),
                                            'trans_type2_perc' : float(result.count(2))/float(len(result)),
                                            'trans_type3_perc' : float(result.count(3))/float(len(result)),
                                            'trans_type4_perc' : float(result.count(4))/float(len(result)),
                                            
                                            
                                           }    
                
    pprint.pprint(all_freqdist_data)
        
            
            
            
    
    
    
    


    
####################
# Stats calc
####################
def plot_freqdist_per_scenario_per_freq(all_data,sc_list, metric):
    fig, axarr = plt.subplots(3,int(np.ceil(len(all_data.keys())/3.0)), figsize=(16, 5))
    fig.canvas.set_window_title('plot_freqdist_per_scenario_per_freq')
    axs = axarr.ravel()
    
    # get colors and freq list
    (all_freq_list, colsd) = get_allfreq_list(metric)
    
    width = 0.1
    pos = 1.0
    for ix, each_scenario in enumerate(sc_list):
        sc_data = all_data[each_scenario]        
        pdata = np.empty([len(sc_data.keys()), len(all_freq_list)])
        i=0
        lbl1 = []
        for mifint_freqstr, fdata in sc_data.iteritems():    
            lbl1.append(mifint_freqstr)        
            pdata[i] = np.array([fdata[metric]['counter'][f] for f in all_freq_list])
            i+=1
        
        im = axs[ix].pcolormesh(pdata, cmap='Greys', vmin=0, vmax=np.amax(pdata), edgecolor='black', linestyle=':', lw=1.0)
        fig.colorbar(im, ax=axs[ix])
        axs[ix].set_title(each_scenario)
        
        axs[ix].set_xticks(np.arange(1,len(pdata[0])+1))
        axs[ix].set_xticklabels(all_freq_list, rotation=90, fontsize=10)
        
        axs[ix].set_yticks(np.arange(len(pdata)))
        axs[ix].set_yticklabels(lbl1, fontsize=10)
    
    
    
def OLD_plot_freqstats_per_scenario_per_freq(all_data, sc_list, fmetric, smetric, mifint_freq_list=None):
    fig, axarr = plt.subplots(2,int(np.ceil(len(all_data.keys())/2.0)), figsize=(16, 5))
    fig.canvas.set_window_title('plot_freqstats_per_scenario_per_freq - ' + fmetric + " - " + smetric)
    axs = axarr.ravel()
    
    # get colors and freq list
    (all_freq_list, colsd) = get_allfreq_list(fmetric)
    
    width = 0.3
    pos = 1.0
    for ix, each_scenario in enumerate(sc_list):
        sc_data = all_data[each_scenario]        
        ydata = [] 
        xlbl = []       
        
        if mifint_freq_list == None:
            for mifint_freqstr, fdata in sc_data.iteritems():        
                ydata.append(fdata[fmetric][smetric])
                xlbl.append(mifint_freqstr)
        else:
            for mifint_freqstr in mifint_freq_list:
                if mifint_freqstr in sc_data: 
                    fdata = sc_data[mifint_freqstr]        
                    ydata.append(fdata[fmetric][smetric])
                    xlbl.append(mifint_freqstr)
                else:
                    pass
                  
        #y_norm = _normalise_list_sum(ydata)
        y_norm = ydata 
        xdata = np.arange(1,len(y_norm)+1)
        b = axs[ix].bar(xdata, y_norm, width, color='blue')
        
        axs[ix].set_title(each_scenario)
        #axs[ix].set_ylim([np.min(ydata)*0.75, np.max(ydata)*1.05])
        axs[ix].set_xticks(xdata)
        axs[ix].set_xticklabels(xlbl, rotation=35, fontsize=10)
        





def plot_freqstats_per_scenario_per_freq(all_data, sc_list, fmetric, smetric, mifint_freq_list, ylbl):
    print "plot_freq_time_in_state :: Enter"
    fig, axarr = plt.subplots(1,1, figsize=(5, 5))
    fig.canvas.set_window_title('plot_freqstats_per_scenario_per_freq - ' + fmetric + " - " + smetric)
    
    fsize=15
    width = 0.15
    scatter_colors = plt.get_cmap('Blues')(np.linspace(0, 1.0, len(mifint_freq_list)))
    cols = [rlbench_test_mapping[f][1] for f in mifint_freq_list]
    print len(scatter_colors)
    
    all_max = 0  
    for ix, each_scenario in enumerate(sc_list):
        print each_scenario
        sc_data = all_data[each_scenario]        
        ydata = [] 
        xlbl = []
        label=each_scenario       
                
        for mifint_freqstr in mifint_freq_list:
            if mifint_freqstr in sc_data: 
                fdata = sc_data[mifint_freqstr]        
                ydata.append(fdata[fmetric][smetric])
                xlbl.append(mifint_freqstr)
            else:
                pass    
        
        if np.max(ydata) > all_max: 
            all_max = np.max(ydata)
        else:
            pass
        
        
        ind = (ix) + (width * np.arange(0,len(mifint_freq_list)))
        rect = axarr.bar(ind, ydata, width, color=cols)    
        
    # legend
    rect_lbl_list = [rlbench_test_mapping[f][0] for f in mifint_freq_list]    
    rects_list = []
    for ix, each_rect in enumerate(rect_lbl_list):
        rec = patches.Rectangle( (0.72, 0.1), 0.2, 0.6, facecolor=cols[ix])
        rects_list.append(rec)
    
    leg = plt.figlegend( rects_list, rect_lbl_list, loc = 'upper center', 
                         ncol=len(rects_list)/2, labelspacing=0. , fontsize=13, handletextpad=0.2,
                         frameon=False, )
   
    leg.get_frame().set_facecolor('#FFFFFF')
    leg.get_frame().set_linewidth(0.0)
    leg.draggable()
    
    xticks = np.arange(0, len(sc_list)) + (width*(len(mifint_freq_list)/2.))
    axarr.set_xticks(xticks)
    sc_lbls = [roylongbottom_microbench_list_renames[s] for s in sc_list]
    axarr.set_xticklabels(sc_lbls, fontsize=fsize, rotation=15)
    axarr.set_xlim([-1*width, (len(sc_list))])
    axarr.set_ylim([0, all_max*1.01])
    axarr.set_ylabel(ylbl, fontsize=fsize)
    axarr.tick_params(axis='y', which='major', labelsize=fsize)
    
    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    
    plt.subplots_adjust(top=0.88, left=0.14, right=.99, bottom=0.12, hspace=0.20, wspace=0.20)
    




def plot_freq_time_in_state(sc_list, metric, TMP_MIF_FREQ = "default", TMP_INT_FREQ = "default"):
    
    f, axarr = plt.subplots(2,8, sharex=True, sharey=True, figsize=(16, 10))
    f.canvas.set_window_title('plot_time_in_state -'+metric)
    axarr = axarr.ravel()
    
    
    
    # get colors and freq list
    (freq_list, colsd) = get_allfreq_list(metric)
    
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

def _counter_mode(lst):
    data = collections.Counter(lst)
    return data.most_common(1)[0][0]


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


#plot_freq_time_in_state(scenario_list, 'gpu_freq', TMP_MIF_FREQ='400000', TMP_INT_FREQ='400000')

mifint_freq_list = [ 
                     "default-default",                      
                     #"test0-test0",                             
                     "test2-test2",
                     "test3-test3",
                     "test1-test1",
                     "test4-test4",
                     "test5-test5",                                            
                     ]


freqdist_alldata =  get_freqstats_per_scenario_per_test(roylongbottom_microbench_list, ['cpu_freq', 'gpu_freq', 'bus_mif_freq', 'bus_int_freq'])
#plot_freqdist_per_scenario_per_freq(freqdist_alldata,roylongbottom_microbench_list, 'cpu_freq')
#plot_freqstats_per_scenario_per_freq(freqdist_alldata, roylongbottom_microbench_list, 'mif_bus_freq', 'sum', mifint_freq_list=mifint_freq_list)
plot_freqstats_per_scenario_per_freq(freqdist_alldata, roylongbottom_microbench_list, 'cpu_freq', 'transitions', mifint_freq_list, '# of CPU-freq transitions')
#plot_freqstats_per_scenario_per_freq(freqdist_alldata, roylongbottom_microbench_list, 'cpu_freq', 'sum', mifint_freq_list, 'Total CPU-freq (Hz)')
#plot_freqstats_per_scenario_per_freq(freqdist_alldata, 'cpu_freq', 'transitions')

#get_metric_slope_analysis(scenario_list)

plt.show()
print "-- Finished --"
