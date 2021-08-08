

import os
import csv
import collections
import numpy as np

import threading

##############################################################
# save activation for data-based normalization
##############################################################
# distribution of activation - neuron-wise or channel-wise?
def save_act_stat(self):

    #path_stat='/home/sspark/Projects/05_SNN/stat/'
    #path_stat='./stat/'
    path_stat=self.conf.path_stat
    #f_name_stat='act_n_train_after_w_norm_max_999'
    #f_name_stat='act_n_train'
    f_name_stat_pre=self.conf.prefix_stat
    stat_conf=['max_999']
    #stat_conf=['max','mean','max_999','max_99','max_98']
    #stat_conf=['max_95','max_90']
    #stat_conf=['max','mean','min','max_75','max_25']
    f_stat=collections.OrderedDict()
    #wr_stat=collections.OrderedDict()

    #
    threads=[]

    for idx_l, l in enumerate(self.list_layer_name_write_stat):
        for idx_c, c in enumerate(stat_conf):
            key=l+'_'+c

            f_name_stat = f_name_stat_pre+'_'+key
            f_name=os.path.join(path_stat,f_name_stat)
            #f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'w')
            #f_stat[key]=open(path_stat'/'f_name_stat)
            #print(f_name)

            f_stat[key]=open(f_name,'w')
            #wr_stat[key]=csv.writer(f_stat[key])


            #for idx_l, l in enumerate(self.list_layer_name_write_stat):
            threads.append(threading.Thread(target=write_stat, args=(self,f_stat[key], l, c)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def write_stat(self, f_stat, layer_name, stat_conf_name):
    print('---- write_stat ----')

    l = layer_name
    c = stat_conf_name
    s_layer=self.dict_stat_w[l].numpy()

    _write_stat(self,f_stat,s_layer,c)


def _write_stat(self, f_stat, s_layer, conf_name):
    print('stat cal: '+conf_name)

    if conf_name=='max':
        stat=np.max(s_layer,axis=0).flatten()
        #stat=tf.reshape(tf.reduce_max(s_layer,axis=0),[-1])
    elif conf_name=='max_999':
        stat=np.nanpercentile(s_layer,99.9,axis=0).flatten()
    elif conf_name=='max_99':
        stat=np.nanpercentile(s_layer,99,axis=0).flatten()
    elif conf_name=='max_98':
        stat=np.nanpercentile(s_layer,98,axis=0).flatten()
    else:
        print('stat confiugration not supported')

    print('stat write - {}'.format(f_stat.name))
    wr_stat=csv.writer(f_stat)
    wr_stat.writerow(stat)
    f_stat.close()
