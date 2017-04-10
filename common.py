import csv
import numpy as np
#from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL


CPU_FREQS_KHZ = [1600000,1500000,1400000,1300000,1200000,1100000,1000000,900000,800000,600000,550000,500000,450000,400000,350000,300000,250000] 
GPU_FREQS_MHZ = [532,480,350,266,177] 
INT_FREQS_KHZ = [800000, 700000, 650000, 600000, 550000, 400000, 267000, 200000, 160000, 100000, 50000]
MIF_FREQS_KHZ = [800000, 400000, 200000, 100000]


NUM_CPU_CORES=4

# colors based on frequency (MIF/INT)
DEFAULT_COL_LIST ={                    
                    "default-default" : '#636363',
                    "800000-800000" : '#a50f15',
                    "800000-700000" : '#de2d26',
                    "800000-600000" : '#fb6a4a',
                    "800000-400000" : '#fcae91',
                    "800000-200000" : '#fee5d9',
                    "400000-400000" : '#2171b5',
                    "400000-200000" : '#6baed6',
                    "400000-100000" : '#eff3ff',
                    "400000-50000" : '#bdd7e7',                    
                    "200000-200000" : '#31a354',
                    "200000-100000" : '#e5f5e0',
                    "200000-50000" : '#a1d99b',                                        
                    "100000-50000" : '#feb24c',
                    }


all_mifint_freqs_macroworkload = [
                                  "default-default",
                                  "800000-800000",
                                  "800000-700000",
                                  "800000-600000",                                  
                                  "800000-200000",
                                  "400000-400000",
                                  "400000-50000",
                                  "200000-200000",
                                  "200000-50000",                                  
                                  "100000-50000",
                                  ]
                                  

rlbench_test_mapping = {
                             # code: [lbl, col, mifintstr]
                             
                             "default-default": ["default", 'darkgray', "default-default"],                      
                             "test0-test0": ["fixedAll", 'green', "400-160"],                             
                             "test2-test2": ["400-160", 'royalblue', "400-160"],
                             "test3-test3": ["800-800", 'skyblue', "800-800"],
                             "test1-test1": ["randMIF", '#fcae91', "RND-160"],
                             "test4-test4": ["randINT", '#fb6a4a', "800-RND"],
                             "test5-test5": ["randMem", '#cb181d', "RND-RND"],  
                             
                             }



scenario_list =[ 
                # idle or almost idle 
                "idle0", "idle1", 
                "launcher0",
                
                # multimedia
                "ffmpeg0",  # background
                "vlcplayer0",
                "music0",   # background
                "camera0",
                "camera1",
                
                # communication/social media
                "line0","line1", "line2",
                "facebook0", "facebook1",
                
                # browsing
                "chrome0",
                
                # download
                "ftp0",
                
                # browsing
                "game0",
                   
            ]


markers_and_cols_per_scenario = {
                                 
                                 # idle or almost idle 
                                "idle0" : ['2', 'blue'], 
                                "idle1" : ['2', 'saddlebrown'],
                                
                                "launcher0" : ['1', "cyan"],
                                
                                # multimedia
                                
                                
                                "vlcplayer0" : ['s', "orange"],
                                
                                   
                                "camera0" : ['.', 'blue'],
                                "camera1" : ['.', 'red'],
                                
                                # communication/social media
                                "line0" : ['*', 'blue'],
                                "line1" : ['*', 'red'],
                                "line2" : ['*', 'green'],
                                
                                "facebook0" : ['x', 'blue'],
                                "facebook1" : ['x', 'saddlebrown'],
                                
                                # browsing
                                "chrome0" : ['3', 'green'],
                                
                                # download
                                "ftp0" : ['+', "gold"],
                                "music0" : ['+', "red"],
                                "ffmpeg0" : ['+', "lime"],
                                
                                # browsing
                                "game0" : ['+', 'k'],
                                 
                                 
                                 } 


microbench_list = [                   
                   'spec_bzip2', 'spec_gobmk',
                   'spec_hmmer', 'spec_mcf',
                   'spec_sjeng',
                   ]


roylongbottom_microbench_list = [
                                 'rlbench_mprndmemi',
                                 'rlbench_mpbusspd2',
                                 'rlbench_mpdhryi',
                                 'rlbench_mpmflops2i',                                 
                                 ]


roylongbottom_microbench_list_renames = {
                                 'rlbench_mprndmemi' : 'mprndmemi',
                                 'rlbench_mpbusspd2' : 'mpbusspd2',
                                 'rlbench_mpdhryi' : 'mpdhryi',
                                 'rlbench_mpmflops2i' : 'mpmflops2i',                                
                                 }



scenario_list_fps = [
                        # idle or almost idle 
                        "launcher0",
                        
                        # multimedia                        
                        "vlcplayer0",                        
                        "camera0", "camera1",
                        
                        # communication/social media
                        "line0","line1", "line2",
                        "facebook0", "facebook1",
                        
                        # browsing
                        "chrome0",
                        
                        # browsing
                        "game0",
                        
                     ]



target_metrics_order =        [
                         
                        # related to bus monitor - bandwidth
                        "bw_mfc0", "bw_mfc1", 
                        "bw_isp0", "bw_isp1",
                        #"bw_gen", "bw_fsys",
                        "bw_mem0_0", "bw_mem1_0",
                        "bw_mem0_1", "bw_mem1_1",           
                        "bw_disp1", 
                        #"bw_total",
                        
                        # related to bus monitor - saturation
                        "sat_mfc0", "sat_mfc1", 
                        "sat_isp0", "sat_isp1",
                        #"sat_gen", "sat_fsys",
                        "sat_mem0_0", "sat_mem1_0",
                        "sat_mem0_1", "sat_mem1_1",           
                        "sat_disp1", 
                        #"sat_total",

                                                
                        # related to cpu/gpu
                        "cpu_util", "gpu_util",
                        
                        # related to frequencies
                        "cpu_freq", "gpu_freq",
                        "bus_mif_freq", "bus_int_freq"                      
                        
                        ]


reduced_target_metrics_order =[                               
                               "sat_total",
                               "sat_cost",
                               
                               # related to cpu/gpu
                               "cpu_util", "gpu_util",
                               "cpu_cost", "gpu_cost",
                        
                               # related to frequencies
                               "cpu_freq", "gpu_freq",
                               "bus_mif_freq", "bus_int_freq",
                               ]

reduced_target_metrics_order_nogpu =[                               
                               "sat_total",
                               "sat_cost",
                               
                               # related to cpu/gpu
                               "cpu_util", 
                               "cpu_cost", 
                        
                               # related to frequencies
                               "cpu_freq", 
                               "bus_mif_freq", "bus_int_freq",
                               ]

reduced_target_metrics_order_2 =[                               
                               "sat_cost",
                               
                               # related to cpu/gpu
                               #"cpu_util", "gpu_util",
                               "cpu_cost", "gpu_cost",
                        
                               # related to frequencies
                               "cpu_freq", "gpu_freq",
                               "bus_mif_freq", "bus_int_freq",
                               ]

reduced_metrics_onlyfreqs_order =[  
                               # related to frequencies
                               "cpu_freq", "gpu_freq",
                               "bus_mif_freq", "bus_int_freq",
                               
                               "sat_total", "sat_cost"
                               ] 

reduced_metrics_onlyfreqs_order_2 = [  
                               # related to frequencies
                               "cpu_freq", "gpu_freq",
                               "bus_mif_freq", "bus_int_freq",
                               ]


 

def rename_metrics(met_name):
    new_met_name = met_name
    if met_name == "sat_total":
        new_met_name = "mem_util"
    
    if met_name == "sat_cost":
        new_met_name = "mem_cost"
    
    #if met_name == "cpu_util_freq":
    #    new_met_name = "cpu_util"
    
    if met_name == "bus_mif_freq":
        new_met_name = "MIF freq"
    
    if met_name == "bus_int_freq":
        new_met_name = "INT freq"
    
    
    return new_met_name



def load_csv(mem_fname, cpugpu_fname):
    NUM_CPU_CORES = 4
    all_metrics_data = {
                        # related to bus monitor - bandwidth
                        "bw_mfc0":[], "bw_mfc1":[], 
                        "bw_isp0":[], "bw_isp1":[],
                        "bw_gen":[], "bw_fsys":[],
                        "bw_mem0_0":[], "bw_mem1_0":[],
                        "bw_mem0_1":[], "bw_mem1_1":[],           
                        "bw_disp1":[], 
                        "bw_total":[],
                        
                        # related to bus monitor - saturation
                        "sat_mfc0":[], "sat_mfc1":[], 
                        "sat_isp0":[], "sat_isp1":[],
                        "sat_gen":[], "sat_fsys":[],
                        "sat_mem0_0":[], "sat_mem1_0":[],
                        "sat_mem0_1":[], "sat_mem1_1":[],           
                        "sat_disp1":[], 
                        "sat_total":[],
                        "sat_cost":[],
                                                
                        # related to cpu/gpu
                        "core0_util":[],"core1_util":[],
                        "core2_util":[],"core3_util":[],                        
                        "cpu_util":[], "gpu_util":[],
                        'cpu_cost':[], "gpu_cost":[],
                        
                        # related to frequencies
                        "cpu_freq":[], "gpu_freq":[],
                        "bus_mif_freq":[], "bus_int_freq":[]
                   }
    
    # get memory related stats
    with open(mem_fname, 'rb') as f:
        reader = csv.reader(f, delimiter=',')        
        for row in reader:
            sampleCount = row[0]
            #print sampleCount    
            
            # bw, freq, sat, duration                    
            all_metrics_data["bw_mfc0"].append(float(row[1]))
            all_metrics_data["bw_mfc1"].append(float(row[5]))
            all_metrics_data["bw_isp0"].append(float(row[9]))
            all_metrics_data["bw_isp1"].append(float(row[13]))
            all_metrics_data["bw_gen"].append(float(row[17]))
            all_metrics_data["bw_fsys"].append(float(row[21]))
            all_metrics_data["bw_mem0_0"].append(float(row[25]))
            all_metrics_data["bw_mem1_0"].append(float(row[29]))
            all_metrics_data["bw_mem0_1"].append(float(row[33]))
            all_metrics_data["bw_mem1_1"].append(float(row[37]))
            all_metrics_data["bw_disp1"].append(float(row[41]))
            
            all_metrics_data["sat_mfc0"].append(float(row[3]))
            all_metrics_data["sat_mfc1"].append(float(row[7]))
            all_metrics_data["sat_isp0"].append(float(row[11]))
            all_metrics_data["sat_isp1"].append(float(row[15]))
            all_metrics_data["sat_gen"].append(float(row[19]))
            all_metrics_data["sat_fsys"].append(float(row[23]))
            all_metrics_data["sat_mem0_0"].append(float(row[27]))
            all_metrics_data["sat_mem1_0"].append(float(row[31]))
            all_metrics_data["sat_mem0_1"].append(float(row[35]))
            all_metrics_data["sat_mem1_1"].append(float(row[39]))
            all_metrics_data["sat_disp1"].append(float(row[43]))
            
            
            # mif, int freqs
            #if MIF_FREQ=="default" and INT_FREQ=="default":
            if len(row)>44:
                all_metrics_data["bus_mif_freq"].append(int(row[45]))
                all_metrics_data["bus_int_freq"].append(int(row[46]))
    
    # calculate totals
    all_metrics_data['sat_total'] =  np.array(all_metrics_data["sat_mfc0"]) + \
                 np.array(all_metrics_data["sat_mfc1"]) + \
                 np.array(all_metrics_data["sat_isp0"]) + \
                 np.array(all_metrics_data["sat_isp1"]) + \
                 np.array(all_metrics_data["sat_gen"]) + \
                 np.array(all_metrics_data["sat_fsys"]) + \
                 np.array(all_metrics_data["sat_mem0_0"]) + \
                 np.array(all_metrics_data["sat_mem1_0"]) + \
                 np.array(all_metrics_data["sat_mem0_1"]) + \
                 np.array(all_metrics_data["sat_mem1_1"]) + \
                 np.array(all_metrics_data["sat_disp1"])
    
    all_metrics_data['bw_total'] =  np.array(all_metrics_data["bw_mfc0"]) + \
                 np.array(all_metrics_data["bw_mfc1"]) + \
                 np.array(all_metrics_data["bw_isp0"]) + \
                 np.array(all_metrics_data["bw_isp1"]) + \
                 np.array(all_metrics_data["bw_gen"]) + \
                 np.array(all_metrics_data["bw_fsys"]) + \
                 np.array(all_metrics_data["bw_mem0_0"]) + \
                 np.array(all_metrics_data["bw_mem1_0"]) + \
                 np.array(all_metrics_data["bw_mem0_1"]) + \
                 np.array(all_metrics_data["bw_mem1_1"]) + \
                 np.array(all_metrics_data["bw_disp1"])
    
    
    
    
    # cpu gpu util and freqs
    core_data = {                      
            cid: {
                      "busy":[],
                      "nicebusy":[],
                      "idle":[],
                      "util":[], 
                      "util_freq":[],
                      "util_freq_norm" : [], # normalised 'util_freq'
                      } for cid in xrange(NUM_CPU_CORES)
                 }            
            
    with open(cpugpu_fname, 'rb') as f:
        reader = csv.reader(f, delimiter=',')                
        for row in reader:
            sampleCount = row[0]
            #count, freq, {busy, nicebusy, idle} - per CPU, gpufreq, gpuutil            
            # bw, freq, sat, duration                    
            all_metrics_data["cpu_freq"].append(float(row[1]))
  
            cid=0
            core_data[cid]["busy"].append(float(row[2]))
            core_data[cid]["nicebusy"].append(float(row[3]))
            core_data[cid]["idle"].append(float(row[4]))
            
            cid=1
            core_data[cid]["busy"].append(float(row[5]))
            core_data[cid]["nicebusy"].append(float(row[6]))
            core_data[cid]["idle"].append(float(row[7]))
            
            cid=2
            core_data[cid]["busy"].append(float(row[8]))
            core_data[cid]["nicebusy"].append(float(row[9]))
            core_data[cid]["idle"].append(float(row[10]))
                
            cid=3
            core_data[cid]["busy"].append(float(row[11]))
            core_data[cid]["nicebusy"].append(float(row[12]))
            core_data[cid]["idle"].append(float(row[13]))
                
            # gpu stuff
            all_metrics_data['gpu_freq'].append(float(row[14])) # in MHz
            all_metrics_data['gpu_util'].append((float(row[15])/256.0)*100.0)        
    
    # convert low-level core stats into proper metrics
    core_data = calc_and_update_cpu_util(core_data, all_metrics_data["cpu_freq"])
    
    all_metrics_data["core0_util"] = core_data[0]['util']
    all_metrics_data["core1_util"] = core_data[1]['util']
    all_metrics_data["core2_util"] = core_data[2]['util']
    all_metrics_data["core3_util"] = core_data[3]['util']
    
    # calculate total (overall) util
    overall_util = np.array([0.0]*len(core_data[0]["util"]))    
    for cid in xrange(NUM_CPU_CORES):        
        overall_util += core_data[cid]["util"]
    
    all_metrics_data["cpu_util"] = overall_util/float(NUM_CPU_CORES)
    
    # cpugpu cost calc
    all_min=0.0
    all_max=100.0*CPU_FREQS_KHZ[0]
    all_metrics_data["cpu_cost"] = np.array(_normalise_list(np.multiply(np.array(all_metrics_data["cpu_util"]) , np.array(all_metrics_data["cpu_freq"])), 
                                                   norm_min=all_min,
                                                   norm_max=all_max))*100.0
                                                   
    all_min=0.0
    all_max=100.0*GPU_FREQS_MHZ[0]    
    all_metrics_data["gpu_cost"] = np.array(_normalise_list(np.multiply(np.array(all_metrics_data["gpu_util"]) , np.array(all_metrics_data["gpu_freq"])),
                                                   norm_min=all_min,
                                                   norm_max=all_max))*100.0
    
    # calc sat total with freq
    all_min=0.0
    all_max=100.0*INT_FREQS_KHZ[0] * MIF_FREQS_KHZ[0]    
    sat_cost = np.array(all_metrics_data['sat_total']) * np.array(all_metrics_data['bus_mif_freq']) * np.array(all_metrics_data['bus_int_freq'])
    sat_cost = np.array(_normalise_list(sat_cost, norm_min=all_min, norm_max=all_max))*100.0
    all_metrics_data["sat_cost"] = sat_cost
    
    mem_util = (np.array(all_metrics_data['sat_mem0_1']) + np.array(all_metrics_data['sat_mem1_1']) + \
                    np.array(all_metrics_data['sat_mem1_0']) + np.array(all_metrics_data['sat_mem0_0']))* np.array(all_metrics_data['bus_mif_freq']) * np.array(all_metrics_data['bus_int_freq'])
                    
    all_metrics_data["mem_cost"] = np.array(_normalise_list(mem_util, norm_min=all_min, norm_max=all_max))
    
    
    return len(all_metrics_data["cpu_freq"]), all_metrics_data



def calc_and_update_cpu_util(data, cpufreq):    
    # calculate core specific util
    for cid in xrange(NUM_CPU_CORES):        
        zeropad = np.array([0.0])
        
        # calculate core specific util
        cbusy = np.array(data[cid]["busy"])
        cnicebusy = np.array(data[cid]["nicebusy"])
        cidle = np.array(data[cid]["idle"])
        
        cbusysub = np.diff(cbusy)
        cbusysub = np.concatenate([zeropad, cbusysub])
        
        cnicebusysub = np.diff(cnicebusy)
        cnicebusysub = np.concatenate([zeropad, cnicebusysub])
        
        cidlesub = np.diff(cidle)
        cidlesub = np.concatenate([zeropad, cidlesub])
        
        cidlesub_plus_busysub = cbusysub + cidlesub
        
        ratio = np.nan_to_num((cbusysub / cidlesub_plus_busysub))
        
        cutil = ratio * 100.0
     
        cutil_freq = ratio * np.array(cpufreq)
                
        data[cid]["util"] = cutil
        data[cid]["util_freq"] = cutil_freq
    
    # calculate normlised CPU core util
    # find global min/max     
    all_min = 0
    all_max = 0
    for cid in xrange(NUM_CPU_CORES):
        cmax = np.max(data[cid]["util_freq"])
        if cmax > all_max:
            all_max = cmax
        else:
            pass
    
    for cid in xrange(NUM_CPU_CORES):        
        normed_util_freq = _normalise_list(data[cid]["util_freq"], 
                                           norm_min=all_min, norm_max=all_max)
        data[cid]["util_freq_norm"] = np.array(normed_util_freq)*100.0
    
    return data
    


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


def mif_int_freqstr_to_tuple(s):
    [mifstr, intstr] =  s.split("-")    
    return [mifstr, intstr]


