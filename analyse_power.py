import sys
import csv
import pprint
import matplotlib
import collections
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')

from common import DEFAULT_COL_LIST


MOV_AVG_WIN_SIZE = 128

BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/MonsoonDataCapture/060317/"

XAXIS_STEP_SIZE = 0.0002


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

######################
#  Data manipulation
######################
def crop_data(data, start=20000, end="def"):
    pass





#################
#  Plotting
#################
def plot_power(data_list, freq_list, scenario="Test", cols = DEFAULT_COL_LIST):
    fig = plt.figure(figsize=(12*1.2, 4.5*1.2))
    fig.canvas.set_window_title('plot_power')
    
    for each_data, each_freq in zip(data_list, freq_list):
        lbl = str(each_freq) + ' Khz'
        each_col = cols[each_freq]     
        #plt.plot(each_data, c=each_col, alpha=0.3, label=lbl)
        #plt.hold(True)
        
        mov_av = movingaverage(each_data, MOV_AVG_WIN_SIZE)[(MOV_AVG_WIN_SIZE/2.0):(MOV_AVG_WIN_SIZE/2.0)*-1]
        sample_size = len(mov_av)
        xdata = np.linspace(0.0, sample_size*XAXIS_STEP_SIZE, sample_size)
        
        plt.plot(xdata, mov_av, c=each_col, label=lbl+"-M.avg.", linewidth=1.0)
        #plt.plot(mov_av, c=each_col, label=lbl+"-M.avg.", linewidth=1.0)
        plt.hold(True)
    
    plt.grid(True)
    
    plt.ylabel("Power (mW)")
    plt.xlabel("Time (s)")
    
    #plt.xlim(0.0, xdata[-1]+0.05)
        
    l = plt.legend(fontsize=12)
    l.draggable()
    
    plt.subplots_adjust(top=0.97, left=0.06, right=0.99, bottom=0.10)
    
    

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
             "pw_mt" : np.sum(data)*(len(data)*XAXIS_STEP_SIZE),
             "pw_dt" : np.sum(data)/(len(data)*XAXIS_STEP_SIZE),
        }  
    
    if print_report==True:  
        print "------------"
        print "{0}, {1}".format(scenario, freq)
        print "------------"     
        pprint.pprint(s) 
        print ""
        
    return s

    
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
SCENARIO_ID = "idle1" 
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
START_SAMPLE_IX = 0
END_SAMPLE_IX = -1

CUSTOM_CROPPING_PARAMS = collections.OrderedDict(
                        [                          
                         ("default-default" , [5000,-5000-2924]),                         
                         ("800000-800000" , [5000,-5000-2879]),
                         ("400000-400000" , [5000,-5000-990]),
                         ("200000-200000" , [5000,-5000-2399]),
                         ("200000-50000" , [5000,-5000-255]),
                         ("400000-50000" , [5000,-5000-1]),                                        
                          ])                                            

DEVFREQ_MIFINT_PAIRS = [str_2_mifint(k) for k in CUSTOM_CROPPING_PARAMS.keys()]

# gather data (csv)
data_list = [] # raw
cropped_data_list = [] # cropped
freq_str_list = []

all_min_sample_size = sys.maxint

for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
    freq_str = '{0}-{1}'.format(mif_freq, int_freq)
    freq_str_list.append(freq_str)
    data_fname = DATA_DIR+'pmdata_mif{0}_int{1}.csv'.format(mif_freq, int_freq)
    
    # get csv data
    (final_time, data) = load_csv(data_fname)
    data_list.append(data)
    data_sample_size = len(data)
    
    print final_time
    
    # used for cropping
    if (data_sample_size < all_min_sample_size):
        all_min_sample_size = data_sample_size
    else:
        pass
    
# start cropping
for each_data, each_freq_str in zip(data_list, freq_str_list):    
    
#     if END_SAMPLE_IX == None:
#         cdata = each_data[START_SAMPLE_IX:all_min_sample_size-1]
#     else:
#         cdata = each_data[START_SAMPLE_IX:END_SAMPLE_IX]
#     
    
    cdata = each_data[CUSTOM_CROPPING_PARAMS[each_freq_str][0]:CUSTOM_CROPPING_PARAMS[each_freq_str][1]]
    cropped_data_list.append(cdata)


# report data
all_power_stats_per_freq = {}
for (each_data, each_freq) in zip(cropped_data_list, freq_str_list):
    all_power_stats_per_freq[each_freq] = report_stats(each_data, each_freq, scenario=SCENARIO_ID)
    
# plot data
plot_power(cropped_data_list, freq_str_list, scenario=SCENARIO_ID, cols=COL_LIST)

# helper
adjust_sample_endings([len(x) for x in cropped_data_list])







plt.show()



        