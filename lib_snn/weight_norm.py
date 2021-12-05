

import os
import csv
import collections
import numpy as np

import threading

##############################################################
# save activation for data-based normalization
##############################################################
# distribution of activation - neuron-wise or channel-wise?
#def save_act_stat(self):
def save_act_stat(self):

    #path_stat='/home/sspark/Projects/05_SNN/stat/'
    #path_stat='./stat/'
    #path_stat=self.conf.path_stat
    #model_dataset=self.conf.model+'_'+self.conf.dataset
    #path_stat=os.path.join(path_stat,model_dataset)


    path_stat = os.path.join(self.path_model,self.conf.path_stat)

    #f_name_stat='act_n_train_after_w_norm_max_999'
    #f_name_stat='act_n_train'
    f_name_stat_pre=self.conf.prefix_stat
    stat_conf=['max_999']
    #stat_conf=['max']
    stat_conf=['max_75', 'max_25']
    #stat_conf=['max_99','max_95','max_90','max_80','max_70','max_60','max_50','max_40','max_30','max_20','max_10']
    #stat_conf=['max_99','max_95','max_90','max_80','max_70']
    #stat_conf=['max_20','max_10']
    #stat_conf=['median', 'mean']
    #stat_conf=['max','mean','max_999','max_99','max_98']
    #stat_conf=['max','mean','min','max_75','max_25']
    #f_stat=collections.OrderedDict()
    #wr_stat=collections.OrderedDict()

    #
    #if not os.path.isdir(path_stat):
        #os.mkdir(path_stat)

    os.makedirs(path_stat,exist_ok=True)
    #os.makedirs(path_stat+'/neuron',exist_ok=True)
    #os.makedirs(path_stat+'/layer',exist_ok=True)

    #
    threads=[]

    #for idx_l, l in enumerate(self.list_layer_name_write_stat):
    for idx_l, l in enumerate(self.layers_record):
        for idx_c, c in enumerate(stat_conf):
            key=l.name+'_'+c

            #f_name_stat = f_name_stat_pre+'_'+key
            f_name_stat = key
            #f_name=os.path.join(path_stat,f_name_stat)
            #f_stat[key]=open(f_name,'w')

            #f_stat[key]=open(path_stat+f_name_stat+'_'+key+'_'+self.conf.model_name,'w')
            #f_stat[key]=open(path_stat'/'f_name_stat)
            #print(f_name)

            #wr_stat[key]=csv.writer(f_stat[key])


            #for idx_l, l in enumerate(self.list_layer_name_write_stat):
            #threads.append(threading.Thread(target=write_stat, args=(self,f_stat[key], l.name, c)))
            threads.append(threading.Thread(target=write_stat, args=(self,path_stat,f_name_stat,l.name, c)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def write_stat(self, path_stat, f_stat_name, layer_name, stat_conf_name):
    print('---- write_stat ----')

    s_layer=self.dict_stat_w[layer_name].numpy()

    _write_stat(self,path_stat,f_stat_name,s_layer,stat_conf_name)


def _write_stat(self, path_stat, f_stat_name, s_layer, conf_name):
    print('stat cal: '+conf_name)

    f_name=os.path.join(path_stat,f_stat_name)
    f_stat = open(f_name,'w')

    if conf_name=='max':
        stat=np.max(s_layer,axis=0).flatten()
        #stat=tf.reshape(tf.reduce_max(s_layer,axis=0),[-1])
    elif conf_name=='max_999':
        stat=np.nanpercentile(s_layer,99.9,axis=0).flatten()
    elif 'max_' in conf_name:
        percentile = int(conf_name.split('_')[1])
        stat=np.nanpercentile(s_layer,percentile,axis=0).flatten()
    elif conf_name == 'median':
        stat = np.median(s_layer, axis=0).flatten()
    elif conf_name == 'mean':
        stat = np.mean(s_layer, axis=0).flatten()
    else:
        print('stat configuration not supported')

    print('stat_layer write - {}'.format(f_name))
    wr_stat_layer=csv.writer(f_stat)
    wr_stat_layer.writerow(stat)
    f_stat.close()
