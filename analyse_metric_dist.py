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
                     load_csv, calc_and_update_cpu_util, _normalise_list, \
                     scenario_list


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
    

plot_dist(all_scenario_data_specific_metric, METRIC)


plt.show()
print "-- Finished --"
