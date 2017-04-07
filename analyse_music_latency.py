import sys
import csv
import pprint
import collections
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 


from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL

#import seaborn.apionly as sns
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"



##################
# PLOTTING
##################
def plot_music_time_per_freq(all_data):    
    cmap = plt.get_cmap('Purples')
    
    fig, axs = plt.subplots(1,1)
    fig.canvas.set_window_title("plot_music_time_per_freq")
    
    sorted_freq_list =  sorted(all_data.keys())
    y_data =   [ np.mean(all_data[f]['tot_writes_diff']) for f in sorted_freq_list]
    y_data_norm =   _normalise_list_sum(y_data)
    
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
    
    
    width = 0.4    
    c = [cmap(0.7)]*(len(sorted_freq_list)-1) + [cmap(1.0)]
    rects1 = axs.bar(x_data, y_data_norm, width, color=c)    

    axs.set_xticks(x_data+(width/2.))
    axs.set_xticklabels(xticklbls, fontsize=10)
    axs.set_title("Normalised Audio Latency", fontsize=12)
    axs.set_xlim(width*1.5, len(x_data)+(width*2))
        
    



##################
# DATA MANIP
##################
def _normalise_list_sum(lst):
    norm = [float(i)/max(lst) for i in lst]
    return norm

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
    

def load_log_music0(fname):            
    data = {
            'last_write':[],
            'bl_writes':[],
            'del_writes':[],
            'tot_writes':[],
            'tot_writes_diff':[]
            }
    with open(fname) as f:
        content = f.readlines()
    
    for each_line in content:
        if "----" in each_line: # ignore
            continue
        else:
            if "total writes" in each_line:
                tot_writes = int(find_between(each_line, "total writes: ", "\n"))
                data['tot_writes'].append(tot_writes)
            
            if "last write occurred" in each_line:
                last_write = int(find_between(each_line, "last write occurred (msecs): ", "\n"))
                data['last_write'].append(last_write)
            
            if "delayed writes" in each_line:
                del_writes = int(find_between(each_line, "delayed writes: ", "\n"))
                data['del_writes'].append(del_writes)
            
    
    data['tot_writes_diff'] = 1.0/np.diff(np.array(data['tot_writes']))
    
    return data


def _time_convert(m_s):    
    mins = int(m_s[0:m_s.index('m')])
    secs = float(m_s[m_s.index('m')+1:])
    tot = (mins*60) + secs
    return tot






##########
# MAIN
##########
if __name__ == "__main__":
    SCENARIO_ID = 'music0'
    DEVFREQ_MIFINT_PAIRS = [k.split("-") for k in CUSTOM_CROPPING_PARAMS_ALL[SCENARIO_ID].keys()]
    DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
    
    data_list = {} # raw    
    freq_str_list = []
    for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
        freq_str = '{0}-{1}'.format(mif_freq, int_freq)
        req_str = '{0}-{1}'.format(mif_freq, int_freq)
        freq_str_list.append(freq_str)        
        data_fname = DATA_DIR + "data_audio-{0}-{1}.txt".format(mif_freq, int_freq)
         
        data_list[freq_str] = load_log_music0(data_fname)
        
        
    pprint.pprint(data_list)    
    plot_music_time_per_freq(data_list)
        
        
    plt.show()
    
    