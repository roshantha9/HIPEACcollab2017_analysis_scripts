import sys
import csv
import pprint
import matplotlib
import numpy  as np
from scipy import stats
import collections
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')

from common import target_metrics_order,reduced_target_metrics_order, \
                     load_csv, calc_and_update_cpu_util, _normalise_list, all_mifint_freqs_macroworkload ,\
                     scenario_list, markers_and_cols_per_scenario, \
                     mif_int_freqstr_to_tuple
                     
from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL

BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

NUM_CPU_CORES = 4
SAMPLING_PERIOD = 200


    
    
####################
# Data manipulation
####################




    
####################
# Stats calc
####################



####################
# Plotting
####################
def plot_dist(all_data, metric_id):
    fig = plt.figure(figsize=(8*1.2, 4*1.2))
    fig.canvas.set_window_title("plot_dist - "+ metric_id)
    
    data = [all_data[s] for s in scenario_list]
    pos = np.arange(len(scenario_list))
    xlbls = scenario_list
    
    plt.boxplot(data, positions=pos)
    
    plt.xticks(pos, xlbls, rotation=15)
    
    
def plot_scatter_allsc_allfreq(scenario_list, scatter_plot=True, line_plot=True):
    
    mean_mem_cost = []
    mean_cpu_cost = []
    mean_gpu_cost = []
    
    mean_all_metrics = {}
    
    
    CPU_METRIC = 'cpu_cost'
    MEM_METRIC = 'sat_cost'
    GPU_METRIC = 'gpu_cost'
    
    for each_sc in scenario_list:        
        print each_sc
        possible_freqs = CUSTOM_CROPPING_PARAMS_ALL[each_sc].keys()
        
        tmp_mem = []
        tmp_cpu = []
        tmp_gpu = []
        tmp_fix = []
        
        for fix, each_f in enumerate(all_mifint_freqs_macroworkload):
            
            #print possible_freqs
            
            if each_f in possible_freqs:   
                #print each_sc, each_f         
                
                miffreq, intfreq = mif_int_freqstr_to_tuple(each_f)
                
                
                
                # get sc data
                DATA_DIR = BASE_DATA_DIR + each_sc + "/"
                cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(miffreq, intfreq)
                mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(miffreq, intfreq)
                (count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)
                
                m = markers_and_cols_per_scenario[each_sc][0]
                c = markers_and_cols_per_scenario[each_sc][1]
                
                mean_cpu_cost.append([fix, np.mean(perfdata[CPU_METRIC]), m, c])
                mean_gpu_cost.append([fix, np.mean(perfdata[GPU_METRIC]), m, c])
                mean_mem_cost.append([fix, np.mean(perfdata[MEM_METRIC]), m, c])
                
                tmp_cpu.append(np.mean(perfdata[CPU_METRIC]))
                tmp_gpu.append(np.mean(perfdata[GPU_METRIC]))                
                tmp_mem.append(np.mean(perfdata[MEM_METRIC]))
                tmp_fix.append(fix)
                
            else:
                pass
            
            
        # populate dicts
        mean_all_metrics[each_sc] = {
                                       'cpu': tmp_cpu,
                                       'gpu': tmp_gpu,
                                       'mem': tmp_mem,
                                       'fix' : tmp_fix,
                                       'marker' : markers_and_cols_per_scenario[each_sc][0],
                                       'col' : markers_and_cols_per_scenario[each_sc][1]
                                       }
            
    
    ##### only scatters ####
    if scatter_plot == True:
        fig, axs = plt.subplots(1,3)
        fig.canvas.set_window_title("plot_scatter_allsc_allfreq")
        axs = axs.ravel()
        
        axi = 0    
        titles = [MEM_METRIC.replace('sat','mem'), CPU_METRIC, GPU_METRIC]
        for each_metric_data in [
                                 mean_mem_cost, mean_cpu_cost,mean_gpu_cost
                                 ]: 
            
            
            x,y, m, c = zip(*each_metric_data)        
            for i, each_x in enumerate(x):            
                axs[axi].scatter([x[i]],[y[i]],marker=m[i], color=c[i], s=90, linewidth=3.5)
                axs[axi].hold(True)
                
            axs[axi].set_title(titles[axi])
            axs[axi].xaxis.grid(False)
            axs[axi].yaxis.grid(False)
            
            axi+=1
            
        # legend
        rect_lbl_list = scenario_list
        cols = [markers_and_cols_per_scenario[s][1] for s in scenario_list]
        markers = [markers_and_cols_per_scenario[s][0] for s in scenario_list]
        artist_list = []
        for ix, each_rect in enumerate(rect_lbl_list):        
            a = plt.Line2D((0,1),(0,0), color=cols[ix], marker=markers[ix], linestyle='', mew=1.5, ms=13, mec=cols[ix])        
            artist_list.append(a)
        
        leg = plt.figlegend( artist_list, rect_lbl_list, loc = 'upper center', 
                             ncol=len(artist_list)/4, labelspacing=0. , fontsize=13,
                             frameon=False, numpoints=1)
        leg.get_frame().set_facecolor('#FFFFFF')
        leg.get_frame().set_linewidth(0.0)
        leg.draggable()
    
    
    ##### line and scatters ####
    if line_plot == True:
        fsize=14
        titles = [CPU_METRIC, GPU_METRIC]
        metrics = ['cpu', 'gpu']
        
        fig, axs = plt.subplots(1,len(metrics), figsize=(12.1, 5.5))
        fig.canvas.set_window_title("plot_line_allsc_allfreq")
        axs = axs.ravel()
        
           
        #titles = [MEM_METRIC.replace('sat','mem'), CPU_METRIC, GPU_METRIC]
        #metrics = ['mem', 'cpu', 'gpu']
        
        for axi, each_met in enumerate(metrics):
            ax_ymax = 0
            for each_sc_k, each_sc_data in mean_all_metrics.iteritems():
                
                m = each_sc_data['marker']
                c = each_sc_data['col']
                x = each_sc_data['fix']
                y = each_sc_data[each_met]
                
                if np.max(y) > ax_ymax: ax_ymax=np.max(y)
                
                print x
                print y
                print "---"
                
                axs[axi].plot(x,y,marker=m, color=c, linewidth=2, linestyle='', mew=2, ms=10, markeredgecolor=c)
                axs[axi].hold(True)
                axs[axi].plot(x,y, color=c, linewidth=1, linestyle='-', alpha=0.2)
                axs[axi].hold(True)
            
            
            axs[axi].set_xlim([-1, len(all_mifint_freqs_macroworkload)])
            axs[axi].set_ylim([-2, ax_ymax*1.05])      
            axs[axi].set_title(titles[axi], fontsize=fsize)
            axs[axi].xaxis.grid(False)
            axs[axi].yaxis.grid(False)
            axs[axi].set_xticks(np.arange(0,len(all_mifint_freqs_macroworkload)))
            axs[axi].set_xticklabels([f.replace('000','').replace('default-','') for f in all_mifint_freqs_macroworkload], rotation=45, fontsize=fsize)
            axs[axi].tick_params(axis='both', labelsize=fsize)
            
        # legend
        rect_lbl_list = scenario_list
        cols = [markers_and_cols_per_scenario[s][1] for s in scenario_list]
        markers = [markers_and_cols_per_scenario[s][0] for s in scenario_list]
        artist_list = []
        for ix, each_rect in enumerate(rect_lbl_list):        
            a = plt.Line2D((0,1),(0,0), color=cols[ix], marker=markers[ix], linestyle='', mew=1.5, ms=13, mec=cols[ix])        
            artist_list.append(a)
        
        leg = plt.figlegend( artist_list, rect_lbl_list, loc = 'upper center', 
                             ncol=len(artist_list)/4, labelspacing=0. , fontsize=fsize,
                             frameon=False, numpoints=1, columnspacing=1.0)
        leg.get_frame().set_facecolor('#FFFFFF')
        leg.get_frame().set_linewidth(0.0)
        leg.draggable()
        
        plt.subplots_adjust(top=0.83, left=0.03, right=0.99, bottom=0.15, wspace=0.14, hspace=0.18)




#################
#    MAIN code
#################
MIF_FREQ = "default"
INT_FREQ = "default"
#MIF_FREQ = 400000
#INT_FREQ = 50000
METRIC = "bus_int_freq"

all_scenario_data_specific_metric = {}
for each_scenario in scenario_list:
    DATA_DIR = BASE_DATA_DIR + each_scenario + "/"

    cpugpu_csv_fname = DATA_DIR + "data_cpugpu-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)
    mem_csv_fname = DATA_DIR + "data_mem-{0}-{1}.csv".format(MIF_FREQ, INT_FREQ)

    (count, perfdata) = load_csv(mem_csv_fname, cpugpu_csv_fname)
    all_scenario_data_specific_metric[each_scenario] = perfdata[METRIC]
    

#plot_dist(all_scenario_data_specific_metric, METRIC)

plot_scatter_allsc_allfreq(scenario_list, scatter_plot=False, line_plot=True)

plt.show()
print "-- Finished --"
