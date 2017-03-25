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


BASE_DATA_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/DataCapture/SystemPerfStats/220217/"

DEFAULT_COL_LIST = [
                    '#e41a1c',
                    '#377eb8',
                    '#4daf4a',
                    '#984ea3',
                    '#ff7f00'
                    ]

SAMPLING_PERIOD = 1000
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
    
    data = {            
            "sf_frames" : [],
            "sf_fps" : [],
            "ffmpeg_fps" : [], # ffmpeg decoding fps
            "ffmpeg_speedup" : [], # ffmpeg decoding speed
            }    
    
    
    if fname_fps!=None:
        with open(fname_fps, 'rb') as f:
            reader = csv.reader(f, delimiter=',')                    
            for row in reader:                
                data['sf_frames'].append(int("0x{0}".format(row[0]), 0))
        
        # calc fps
        fr_diff = np.diff(data['sf_frames'])
        fps = np.concatenate([np.array([0.0]), fr_diff])
        data["sf_fps"] = fps
        
                
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
                speed = find_between(each_line, "speed=", "x")
                
                decoded_fps_list.append(int(fps.strip()))
                speed_list.append(speed)
        
        data['ffmpeg_fps'] = decoded_fps_list
        data['ffmpeg_speed'] = speed_list
        
            
    return (len(data), data)


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




####################
# Plotting related
####################
def plot_fps_data(data_list, freq_list,fname, scenario="Test", cols = DEFAULT_COL_LIST, ylbl="FPS"):   
    fig = plt.figure(figsize=(9*1.2, 4*1.2))
    fig.canvas.set_window_title(fname)
    
    for each_data, each_freq, each_col in zip(data_list, freq_list, cols):
        lbl = str(each_freq) + ' Khz'
        sample_size = len(each_data)
        xdata = np.arange(sample_size)        
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

COL_LIST = [DEFAULT_COL_LIST[cix] for cix in [0,1,2,3]]
SCENARIO_ID = "ffmpeg0" 
DATA_DIR = BASE_DATA_DIR + SCENARIO_ID + "/"
CUSTOM_CROPPING_PARAMS = {                          
                          "800000-800000" : [0,-1],
                          "400000-400000" : [0,-1],
                          "200000-200000" : [0,-1],
                          "100000-50000" : [0,-1],
                          #"200000-50000" : [0,-1],
                          }


DEVFREQ_MIFINT_PAIRS = [
                        # MIF, INT
                        [800000, 800000],
                        [400000, 400000],
                        [200000, 200000],
                        [100000, 50000],
                        #[200000, 50000],
                        ]

# gather data (csv)
data_list = [] # raw
cropped_data_list = [] # cropped
freq_str_list = []

all_min_sample_size = sys.maxint

for (mif_freq, int_freq) in DEVFREQ_MIFINT_PAIRS:
    freq_str = '{0}-{1}'.format(mif_freq, int_freq)
    freq_str_list.append(freq_str)
    data_fname = DATA_DIR + "data_ffmpeg-{0}-{1}.log".format(mif_freq, int_freq)
    
    # get csv data    
    #(count, data) = load_csv(fname_fps=data_fname, fname_ffmpeg=None)
    (count, data) = load_csv(fname_fps=None, fname_ffmpeg=data_fname)
    
    # for fps
    #data_list.append(data['sf_fps'])
    #data_sample_size = len(data['sf_fps'])
    
    # for ffmpeg fps
    data_list.append(data['ffmpeg_fps'])
    data_sample_size = len(data['ffmpeg_fps'])
    
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
fname="plot_ffmpeg-{0}-".format(SCENARIO_ID)
plot_fps_data(cropped_data_list, freq_str_list, fname, scenario=SCENARIO_ID, cols=COL_LIST, ylbl="Decode FPS")


plt.show()

print "-- Finished --"







