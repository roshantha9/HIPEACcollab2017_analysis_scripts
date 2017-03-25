import sys
import csv
import pprint
from datetime import datetime
import matplotlib
import numpy  as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

#import seaborn.apionly as sns
plt.style.use('/home/rosh/Documents/EngD/Work/VidCodecWork/VideoProfilingData/analysis_tools/bmh_rosh.mplstyle')


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/060317/"

from common import DEFAULT_COL_LIST

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


def _check_row_extraction(data):    
    pass

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def load_csv(fname_fps, fname_ffmpeg=None):
    #print fname_fps
    data = {            
            # related to surfaceflinger #
            "timestamp" : [],      # avg. frames per second
            "sf_fps" : [],      # avg. frames per second
            "sf_frames" : [],   # num frames rendered            
            "sf_jank" : [],     # jank (?)
            "sf_mfs" : [],      # max frame seperation (milisec)
            "sf_okt" : [],      # Over KPI Times (The KPI is the used time of one frame) (i.e. number of frames that took over KPI to process)
            "sf_ss" : [],       # smoothness score: SS=(FPS/The target FPS)*50+(KPI/MFS)*10+(1-OKPIT/Frames)*40
            
            "adjusted_fps" :[], # this is the fps calculated based on the time stamp 
            "jank_percent" : [],  # this is jank/frames
            
            "target_fps":None,
            
            
            # for ffmpeg #
            "ffmpeg_fps" : [], # ffmpeg decoding fps
            "ffmpeg_speedup" : [], # ffmpeg decoding speed
            
            }    
    
    # gathering frame rate info
    if fname_fps!=None:
        with open(fname_fps, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for rix, row in enumerate(reader):
                if rix > 0:
                    if "Division by zero" not in row[0]:                
                        data['timestamp'].append(row[2])
                        data['sf_fps'].append(float(row[3]))
                        data['sf_frames'].append(float(row[4]))
                        data['sf_jank'].append(float(row[5]))
                        data['sf_mfs'].append(float(row[6]))
                        data['sf_okt'].append(float(row[7]))
                        data['sf_ss'].append(float(row[8]))
                    else:
                        data['timestamp'].append(None)
                        data['sf_fps'].append(0.0)
                        data['sf_frames'].append(0.0)
                        data['sf_jank'].append(0.0)
                        data['sf_mfs'].append(0.0)
                        data['sf_okt'].append(0.0)
                        data['sf_ss'].append(100.0)
                else:                                       
                    data['target_fps'] = int(row[3][4:]) 
                    
    
        # adjusted fps calc
        adjusted_fps = []
        for ix, ts in enumerate(data['timestamp']):
            if ix == 0: #skip the first one
                pass
            else:                
                if ts != None:
                    ts_nw = ts
                    ts_pr = data['timestamp'][ix-1]
                    frames = float(data['sf_frames'][ix])                    
                    
                    if ts_pr == None:
                        seconds = 1.0
                    else:
                        s = _time_diff(ts_pr, ts_nw)
                        seconds = s if s>0.0 else 1.0
                         
                        calc_fps = float(frames)/float(seconds)
                        adjusted_fps.append(calc_fps)
                else:
                    adjusted_fps.append(0.0)
        
        #pprint.pprint(adjusted_fps)
        data["adjusted_fps"] = adjusted_fps
            
        data["jank_percent"] = np.nan_to_num(np.array(data["sf_jank"])/np.array(data["sf_frames"]))
        pprint.pprint( data["jank_percent"] )
    
    # gathering ffmpeg decoding frame speed info        
    if fname_ffmpeg!=None:
        with open(fname_ffmpeg) as f:
            content = f.readlines()
        
        for each_line in content:
            if "fps=" in each_line:
                new_content=each_line.split("\r")
                break
        
        decoded_fps_list = []
        speed_list = []
        for each_line in new_content:
            if "fps=" in each_line and "fps=0.0" not in each_line:
                fps = find_between(each_line, "fps=", "q=")                
                speed = float(find_between(each_line, "speed=", "x"))
                
                decoded_fps_list.append(int(fps.strip()))
                speed_list.append(float(speed))
        
        data['ffmpeg_fps'] = decoded_fps_list
        data['ffmpeg_speed'] = speed_list
        
            
    return (len(data), data)


    


# format: 2017-03-23 12:16:25
# format: yyyy-mm-dd hh:mm:ss
def _time_diff(tstart, tend):
    start = datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(tend, "%Y-%m-%d %H:%M:%S")
    return (end-start).total_seconds()



####################
# Data manipulation
####################

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
# Plotting related
####################
def plot_fps_data(data_list, freq_list,fname, scenario="Test", cols = DEFAULT_COL_LIST, ylbl="FPS"):   
    fig = plt.figure(figsize=(9*1.2, 4*1.2))
    fig.canvas.set_window_title(fname)
    
    for each_data, each_freq in zip(data_list, freq_list):
        each_col = cols[each_freq]     
        lbl = str(each_freq) + ' Khz'
        sample_size = len(each_data)
        xdata = np.linspace(0.0, sample_size*SAMPLING_PERIOD, sample_size)       
        plt.plot(xdata, each_data, c=each_col, label=lbl)
        plt.hold(True)
        
    plt.grid(True)
    
    plt.ylabel(ylbl)
    plt.xlabel("Time (s)")
    
    #plt.xlim(0.0, xdata[-1]+0.05)
        
    l = plt.legend(fontsize=12)
    l.draggable()
    
    plt.subplots_adjust(top=0.97, left=0.06, right=0.99, bottom=0.10)
    
    



#################
#    MAIN code
#################
if __name__ == "__main__":

    COL_LIST = DEFAULT_COL_LIST
    SCENARIO_ID = "launcher0" 
    DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
    CUSTOM_CROPPING_PARAMS = {                          
                              "default-default" : [0,-1],
                               "800000-800000" : [0,-1],
                               "400000-400000" : [0,-1],
                               "200000-200000" : [0,-1],
    #                           #"100000-50000" : [0,-1],
    #                           #"200000-50000" : [0,-1],
                                #"400000-100000" : [0,-1],
                               "400000-50000" : [0,-1],
                              }
    
    
    DEVFREQ_MIFINT_PAIRS = [
                            # MIF, INT
                            ["default", "default"],
                             [800000, 800000],
                             [400000, 400000],
                             [200000, 200000],
    #                         #[100000, 50000],
    #                         #[200000, 50000],
                             #[400000, 100000],
                             [400000, 50000],
                            ]
    
    # gather data (csv)
    data_list = [] # raw
    cropped_data_list = [] # cropped
    freq_str_list = []
    
    all_min_sample_size = sys.maxint
    
    for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
        freq_str = '{0}-{1}'.format(mif_freq, int_freq)
        freq_str_list.append(freq_str)
        #data_fname = DATA_DIR + "data_ffmpeg-{0}-{1}.log".format(mif_freq, int_freq)
        data_fname = DATA_DIR + "data_fps-{0}-{1}.csv".format(mif_freq, int_freq)
        
        # get csv data    
        (count, data) = load_csv(fname_fps=data_fname, fname_ffmpeg=None)
        #(count, data) = load_csv(fname_fps=None, fname_ffmpeg=data_fname)
        
        # for fps
        metric = 'adjusted_fps'
        data_list.append(data[metric])
        data_sample_size = len(data[metric])    
        # for ffmpeg fps
        #data_list.append(data['ffmpeg_fps'])
        #data_sample_size = len(data['ffmpeg_fps'])
        
        # used for cropping
        if (data_sample_size < all_min_sample_size):
            all_min_sample_size = data_sample_size
        else:
            pass
        
    # start cropping
    for each_data, each_freq_str in zip(data_list, freq_str_list):    
        cdata = each_data[CUSTOM_CROPPING_PARAMS[each_freq_str][0]:CUSTOM_CROPPING_PARAMS[each_freq_str][1]]
        cropped_data_list.append(cdata)
        print (each_freq_str, len(cdata))
        
    # plot fps
    fname="plot_fps-{0}-".format(SCENARIO_ID)
    plot_fps_data(cropped_data_list, freq_str_list, fname, scenario=SCENARIO_ID, cols=COL_LIST, ylbl="Avg. FPS")
    
    
    plt.show()
    
    print "-- Finished --"







