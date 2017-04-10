import sys


from common import scenario_list, roylongbottom_microbench_list, mif_int_freqstr_to_tuple, rlbench_test_mapping
from cropping_params_power import CUSTOM_CROPPING_PARAMS_ALL

HTML_OUTPUT_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/plots/DetailedPlots/"

BASE_FIG_DIR = "/home/rosh/Documents/NTU_Work/HiPEAC_collab/plots/DetailedPlots/"

FIG_OUTPUT_DIR_BUSSTATS = "bus_stats/"
FIG_OUTPUT_DIR_BUSFREQ = "bus_freq/"


FIG_OUTPUT_DIR_CPUSTATS = "cpugpu_stats/"
FIG_OUTPUT_DIR_CPUFREQ = "cpugpu_freq/"


#all_sc_list = roylongbottom_microbench_list
all_sc_list = scenario_list
all_freq_list = CUSTOM_CROPPING_PARAMS_ALL


html_main_str = "<html><body>"


for each_sc in all_sc_list:
    print each_sc
    html_main_str += "<hr style=\"height:5px;border:none;color:#333;background-color:#333;\" />\n"
    html_main_str += "<br><br>\n"
    SCENARIO_ID = each_sc 
    
    html_main_str +="<h1>Workload : {0}</h1>\n".format(each_sc)
    
    for each_mifint_freq in all_freq_list[each_sc]:
        
        [MIF_FREQ, INT_FREQ] = mif_int_freqstr_to_tuple(each_mifint_freq)
        
        
        
        if "rlbenc" in each_sc:
            TESTID = rlbench_test_mapping[each_mifint_freq][0]
            lbl = "{0}--Test:{1}".format(SCENARIO_ID, TESTID)
            html_main_str +="<h2 style=\"color:blue;\">Test:{0}</h2>\n".format(TESTID)
        else:
            lbl = "{0}--MIF-{1}:INT-{2}".format(SCENARIO_ID, MIF_FREQ, INT_FREQ)
            html_main_str +="<h2 style=\"color:blue;\">MIF:{0}, INT:{1}</h2>\n".format(MIF_FREQ, INT_FREQ)
        
                
        # mem related fnames
        fname_mem_bussat = FIG_OUTPUT_DIR_BUSSTATS + "plot_bus-{0}-".format(lbl) +"Saturation" + ".png" 
        fname_mem_busbw = FIG_OUTPUT_DIR_BUSSTATS + "plot_bus-{0}-".format(lbl) +"Bandwidth" + ".png"
        fname_mem_busfreq = FIG_OUTPUT_DIR_BUSFREQ + "plot_bus-{0}-".format(lbl) +"MIFINTFreq" + ".png"
        
        # cpugpu related fnames
        fname_cpugpu_util = FIG_OUTPUT_DIR_CPUSTATS + "plot_cpugpu-{0}-".format(lbl) +"CPUGPUUtil" + ".png"
        fname_cpugpu_freq = FIG_OUTPUT_DIR_CPUFREQ + "plot_cpugpu-{0}-".format(lbl) +"CPUGPUFreq" + ".png"
        
        # append bus stats
        if MIF_FREQ =="default" and INT_FREQ=="default":
            img_html = "<a href=\"{0}\"><img src=\"{0}\" width=\"30%\" height=\"30%\"></a>\n" \
                        "<a href=\"{1}\"><img src=\"{1}\" width=\"30%\" height=\"30%\"></a>\n" \
                        "<a href=\"{2}\"><img src=\"{2}\" width=\"30%\" height=\"30%\"></a><br>\n" \
                        "<a href=\"{3}\"><img src=\"{3}\" width=\"30%\" height=\"30%\"></a>\n" \
                        "<a href=\"{4}\"><img src=\"{4}\" width=\"30%\" height=\"30%\"></a>\n".format(
                                                                                                      fname_mem_bussat, fname_mem_busbw, fname_mem_busfreq,
                                                                                                      fname_cpugpu_util, fname_cpugpu_freq
                                                                                                      )
        else:
            img_html = "<a href=\"{0}\"><img src=\"{0}\" width=\"30%\" height=\"30%\"></a>\n" \
                        "<a href=\"{1}\"><img src=\"{1}\" width=\"30%\" height=\"30%\"></a><br>\n" \
                        "<a href=\"{2}\"><img src=\"{2}\" width=\"30%\" height=\"30%\"></a>\n" \
                        "<a href=\"{3}\"><img src=\"{3}\" width=\"30%\" height=\"30%\"></a>\n".format(
                                                                                                      fname_mem_bussat, fname_mem_busbw,
                                                                                                      fname_cpugpu_util, fname_cpugpu_freq
                                                                                                      )
                        
                        
                        
        
        
        html_main_str += img_html


html_main_str += "</body></html>\n"


# write out to file
html_fname = HTML_OUTPUT_DIR + "busstats_macroworkloads.html"
f = open(html_fname, "w")
f.write(html_main_str)
f.close()



