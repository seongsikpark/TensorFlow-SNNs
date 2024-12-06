
#
import tensorflow as tf

#from absl import flags
#flags = flags.FLAGS
#from config import conf
#from config_common import conf
from absl import flags
conf = flags.FLAGS


from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras import backend as K

#
import collections
import os
import csv
import numpy as np
import pandas as pd
import functools

#
import matplotlib.pyplot as plt

#
import lib_snn

from lib_snn.sim import glb_t
#from lib_snn.sim import glb_plot

from lib_snn import config_glb

########################################
# init (on_train_begin)
########################################



########################################
# preprocessing (on_test_begin)
########################################

########################################
# setting (on_test_begin)
########################################
def preproc(self):
    # print summary model
    #print('summary model')
    if self.total_num_neurons==0:
        cal_total_num_neurons(self)

    if self.conf.verbose:
        self.model.summary()
        print_total_num_neurons(self)

    # initialization
    #if not self.init_done:
    set_init(self)

    #
    preproc_sel={
        'ANN': preproc_ann,
        'SNN': preproc_snn,
    }
    preproc_sel[self.model.nn_mode](self)

    # spike loss
    if False:
    #if True:
        for idx_n, neuron in enumerate(self.model.layers_w_neuron):
            #    #neuron.act.add_loss(lambda: tf.reduce_mean(neuron.act.spike_count_int))
            #    #neuron.act.add_loss(lambda: tf.reduce_sum(neuron.act.spike_count_int))
            if neuron.act.loc=='HID':
                #print(neuron.name)
                #print(neuron)
                #print(idx_n)
                #spike = neuron.act.out
                spike = neuron.act.spike_count_int
                spike = 1e-5 * spike
                #self.model.add_loss(lambda spike: tf.reduce_mean(0.01*spike))
                #self.model.add_loss(functools.partial(tf.reduce_mean,spike))
                neuron.act.add_loss(functools.partial(tf.reduce_mean,spike))
                #neuron.act.add_loss(lambda: tf.reduce_mean(0.001*neuron.act.out)

    #self.init_done = True


# reset on test begin
def reset(self):
    for metric in self.model.accuracy_metrics:
        if isinstance(metric,compile_utils.MetricsContainer):
            metric.reset_state()

    if conf.mode=='inference':
        spike_count_epoch_init(self)

    self.model.reset()


#
def reset_batch_ann(self):
    pass

def reset_batch_snn(self):
    # TODO: move to model.py
    self.model.reset_snn()
    #self.model.reset_snn_neuron()
    reset_snn_time_step(self)

    if conf.calibration_vmem:
        lib_snn.calibration.vmem_calibration(self)

#
def set_init(self):
    #print('SNNLIB callback initialization')

    # check
    snn_condition_check(self)

    #
    #if self.model.nn_mode=='ANN':
        #init_ann_run(self)
    #elif self.model.nn_mode=='SNN':
        #init_snn_run(self)

    self.model.init(self.model_ann)

    reset_snn_time_step(self)

    # quantization fine tuning
#    if self.conf.fine_tune_quant:
#        #if False:
#        for layer in self.model.layers_w_act:
#            stat=lib_snn.calibration.read_stat(self,layer,'max_999')
#            stat_max = tf.reduce_max(stat)
#            #layer.quant_max = tf.Variable(stat_max,trainable=False,name='quant_max')
#            layer.quant_max.assign(stat_max)
#            #layer.quant_max = tf.constant(stat_max,shape=[],name='quant_max')
#            print('proproc_ann')
#            print(layer.name)
#            print(layer.quant_max)
#


    #self.init_done=True



#
def snn_condition_check(self):
    pass
    #assert (not self.conf.binary_spike) or (self.conf.binary_spike and (self.conf.n_init_vth==1.0)), 'vth should be 1.0 in binary_spike mode'

#
#def init_snn_run(self):
    #self.model.init_snn(self.model_ann)
#
    ##self.model.set_layers_w_neuron()
#
    ## spike max pool setup
    ##self.model.spike_max_pool_setup()



#
def preproc_ann(self):
    ##self.print_model_conf()
    preproc_ann_norm(self)

    # surrogate DNN model for training SNN with temporal information
    if self.conf.f_surrogate_training_model:
        assert False
        #self.preproc_surrogate_training_model()




#
def preproc_snn(self):

    #
    preproc_ann_to_snn(self)

#
def preproc_ann_norm(self):
    if not self.bn_fusion_done:
        if self.conf.f_fused_bn:
            bn_fusion(self)
        self.bn_fusion_done = True

    # data-based weight normalization
    if not self.w_norm_done:
        if self.conf.f_w_norm_data:
            w_norm_data(self)
        self.w_norm_done=True


    #if self.f_vth_set_and_norm:
    #    lib_snn.calibration.vth_set_and_norm(self)

    #print(self.model.get_layer('conv1').bias)
    ## for debug
    #self.model.bias_disable()
    #print(self.model.get_layer('conv1').bias)

#
def preproc_ann_to_snn(self):
    if self.conf.verbose:
        print('preprocessing: ANN to SNN')

    if not self.bn_fusion_done:
        if self.conf.f_fused_bn or ((self.model.nn_mode == 'ANN') and (self.conf.f_validation_snn)):
            bn_fusion(self)
        self.bn_fusion_done = True

    # data-based weight normalization
    if not self.w_norm_done:
        if self.conf.f_w_norm_data:
            w_norm_data(self)
        self.w_norm_done=True

    #
    #if not self.set_leak_const_done:
        #set_leak_const(self)
        #self.set_leak_const_done=True

    #
    #if self.f_vth_set_and_norm:
    #    lib_snn.calibration.vth_set_and_norm(self)
    #    #lib_snn.calibration.weight_calibration_act_based(self)

    # calibration
    #if self.calibration:
    if not self.calibration_static_done:
        calibration_static(self)


    if False:

        print(self.run_for_calibration)
        print(self.calibration_static_done)
        print(self.calibration_act_based_done)


        if not self.calibration_static_done:
            calibration_static(self)

        #elif self.run_for_calibration and self.calibration_static_done and (not self.calibration_act_based_done):
        #if self.run_for_calibration and self.calibration_static_done and (not self.calibration_act_based_done):
        elif self.calibration_static_done and (not self.calibration_act_based_done):
        #if self.calibration_static_done and (not self.calibration_act_based_done):
            calibration_act_based(self)

        #elif self.run_for_calibration and self.calibration_static_done and self.calibration_act_based_done and (not self.calibration_post_done):
        #elif self.calibration_static_done and self.calibration_act_based_done and (not self.calibration_post_done):
        if self.calibration_static_done and self.calibration_act_based_done and (not self.calibration_post_done):
        #if self.run_for_calibration and self.calibration_static_done and self.calibration_act_based_done and (not self.calibration_post_done):
            calibration_act_based_post(self)

        #else:
        #    assert False


#
def calibration_static(self):
    #
    if self.conf.calibration_vmem_ICLR:
        lib_snn.calibration.vmem_calibration_ICLR(self)

    self.calibration_static_done = True

#
def calibration_static_old(self):

    if self.model.nn_mode=='SNN':
        if self.conf.calibration_vth:
            #lib_snn.calibration.vth_calibration_stat(self,f_norm,stat)
            lib_snn.calibration.vth_calibration_stat(self)
            #lib_snn.calibration.vth_calibration_manual(self)

        if self.conf.vth_toggle:
            lib_snn.calibration.vth_toggle(self)

    #
    if self.conf.calibration_weight:
        lib_snn.calibration.weight_calibration(self)
    #    #lib_snn.calibration.weight_calibration_inv_vth(self)

    #
    if self.conf.calibration_bias:
        lib_snn.calibration.bias_calibration(self)


    #
    #lib_snn.calibration.vth_calibration(self)

    self.calibration_static_done = True

def calibration_act_based(self):
    if self.conf.verbose:
        print('calibration_act_based')

    if self.conf.calibration_weight_act_based:
        lib_snn.calibration.weight_calibration_act_based(self)

    self.calibration_act_based_done = True


def calibration_act_based_post(self):
    if self.conf.verbose:
        print('calibration_act_based_post')

    #assert tf.math.logical_not(tf.math.reduce_all(self.conf.calibration_bias,self.conf.calibration_bias_ICLR_21,self.conf.calibration_bias_ICML_21))
    assert tf.math.logical_not(tf.math.logical_and(self.conf.calibration_vmem,self.conf.calibration_vmem_ICML_21))


    if self.conf.calibration_bias_ICLR_21:
        lib_snn.calibration.bias_calibration_ICLR_21(self)

    if self.conf.calibration_bias_ICML_21:
        lib_snn.calibration.bias_calibration_ICML_21(self)

    if self.conf.calibration_vmem_ICML_21:
        lib_snn.calibration.vmem_calibration_ICML_21(self)


    if self.conf.vth_toggle:
        lib_snn.calibration.vth_toggle(self)

    self.calibration_post_done=True



########################################
# reset (on_test_batch_begin and on_train_batch_begin)
########################################
def preproc_batch(self):
    #print('on_test_batch_begin')

    # reset
    reset_batch_sel={
        'ANN': reset_batch_ann,
        'SNN': reset_batch_snn,
    }
    reset_batch_sel[self.model.nn_mode](self)

    #
    preproc_batch_sel={
        'ANN': preproc_batch_ann,
        'SNN': preproc_batch_snn,
    }
    preproc_batch_sel[self.model.nn_mode](self)

def preproc_batch_ann(self):
    pass

def preproc_batch_snn(self):
    pass
    #reset_snn_sample(self)

def preproc_epoch_train(self,epoch):

    self.epoch = epoch

    #
    preproc_epoch_sel={
        'ANN': preproc_epoch_train_ann,
        'SNN': preproc_epoch_train_snn,
    }
    preproc_epoch_sel[self.model.nn_mode](self,epoch)



def preproc_epoch_train_ann(self, epoch):
    pass


def preproc_epoch_train_snn(self, epoch):
    spike_count_epoch_init(self)

    # test
    #if epoch % 100==0 and epoch != 0:
    ##if epoch>0:
        #for layer in self.model.layers_w_neuron:
            #layer.act.reg_spike_out_const.__mul__(10)



#
def reset_snn_sample(self):
    assert False
    reset_snn_time_step(self)
    self.reset_snn_neuron()
    #self.snn_output = np.zeros((self.num_accuracy_time_point,)+self.list_neuron['fc1'].get_spike_count().numpy().shape)
    self.snn_output.assign(tf.zeros((self.num_accuracy_time_point,)+tuple(self.snn_output_neuron.dim)))
    self.count_accuracy_time_point=0

#
def reset_snn_time_step(self):
    glb_t.reset()


########################################
# (on_test_batch_end)
########################################
def postproc_batch_test(self, batch, logs):
    #print('postproc_batch_test')


    if self.conf.mode=='inference':

        #
        spike_count_batch_end(self)

        # TODO: need?
        if self.model.en_record_output:
            collect_record_output(self)

        if self.run_for_vth_search:
            #lib_snn.calibration.weight_calibration_act_based(self)
            lib_snn.calibration.vth_search(self)

        if self.run_for_calibration_ML:
            #if self.conf.calibration_bias_ICML_21:
            #if not self.vth_search_done:
            #    lib_snn.calibration.weight_calibration_act_based(self)
                #self.vth_search_done=True
            lib_snn.calibration.bias_calibration_ICML_21(self)


        #
        if False:
            print('non zero ratio in ann act (postproc_batch)')
            #for layer in self.layers_w_kernel:

            if not hasattr(self,'nonzero_ratios'):
                self.nonzero_ratios = collections.OrderedDict()

            for layer in self.model.layers_w_kernel:
                self.nonzero_ratios[layer.name] = []

            for layer in self.model.layers_w_kernel:
                non_zero = tf.math.count_nonzero(layer.record_output, dtype=tf.float32)
                non_zero_r = non_zero / tf.cast(tf.reduce_prod(layer.record_output.shape),tf.float32)

                non_zero_r_m = tf.reduce_mean(non_zero_r)

                self.nonzero_ratios[layer.name].append(non_zero_r_m)

                #print(layer.name)
                #print(non_zero_r_m)


            #
            print('')
            print('mean activation, kernel, bias')
            for layer in self.model.layers_w_kernel:

                n_m = tf.reduce_mean(self.nonzero_ratios[layer.name])
                a_m = tf.reduce_mean(layer.record_output)
                k_m = tf.reduce_mean(layer.kernel)
                b_m = tf.reduce_mean(layer.bias)

                print('{:<8}: nonzero - {:.4f}, act - {:.4f}, kernel - {:.4f}, bias - {:.4f}'.format(layer.name,n_m,a_m,k_m,b_m))


        # early stop inference
        if self.conf.early_stop_search:
            #print(self.model)
            idx_acc = self.model.metrics_names.index('acc')
            acc=self.model.metrics[idx_acc].result().numpy()
            #print('debug early stop inference')
            #print(acc)
            #print(self.conf.early_stop_search_acc)

            if acc < self.conf.early_stop_search_acc:
                assert False


        # regularization spike - feature map (accumulated spikes) visulaization
        #if False:
        if conf.reg_spike_vis_fmap_sc:
            visualization_fmap_spike(self,batch)


#
def visualization_fmap_spike(self,batch):
    import keras
    import matplotlib.pyplot as plt
    import numpy as np

    from config_snn_training_WTA_SNN import mode

    save_imgs = False

    fm = []
    layer_names = []

    for layer in self.model.layers_w_neuron:
        if isinstance(layer.act, lib_snn.neurons.Neuron):
            fm.append(layer.act.spike_count_int)
            layer_names.append(layer.name)

    # a = fm[13]
    # plt.matshow(a[0,:,:,0])

    images_per_row = 16
    img_idx = 0
    layer_idx = 0

    # display_grid_h = np.zeros((4,4))
    # plt.figure(figsize=(display_grid_h.shape[1], display_grid_h.shape[0]))

    plot_hist = False
    if plot_hist:
        figs_h, axes_h = plt.subplots(4, 4, figsize=(12, 10))

    # only conv1
    fm = [fm[1]]
    layer_names = [layer_names[1]]

    #
    for img_idx in range(0, 100):
        for layer_name, layer_fm in zip(layer_names, fm):
            n_features = layer_fm.shape[-1]
            size = layer_fm.shape[1]

            n_cols = n_features // images_per_row
            if n_cols == 0:  # n_in
                continue

            if len(layer_fm.shape) == 2:  # fc_layers
                continue

            display_grid = np.zeros(((size + 1) * n_cols - 1, images_per_row * (size + 1) - 1))

            #
            for col in range(n_cols):
                for row in range(images_per_row):
                    #
                    channel_index = col * images_per_row + row
                    channel_image = layer_fm[img_idx, :, :, channel_index].numpy().copy()

                    # normalization
                    if channel_image.sum() != 0:
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype("uint8")

                    display_grid[
                    col * (size + 1):(col + 1) * size + col,
                    row * (size + 1):(row + 1) * size + row] = channel_image

            #
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.axis("off")

            plt.imshow(display_grid, aspect="auto", cmap="viridis")

            # channel intensity
            # image
            stat_image = False
            if stat_image:
                # one image
                channel_image = layer_fm[img_idx, :, :, :].numpy().copy()
            else:
                ## batch
                channel_image = layer_fm.numpy().copy()

            channel_intensity = tf.reduce_mean(channel_image, axis=[0, 1])

            # display_grid_h[layer_idx//4, layer_idx%4] = channel_intensity

            if plot_hist:
                axes_h[layer_idx // 4, layer_idx % 4].hist(tf.reshape(channel_intensity, shape=-1), bins=100)

                ci_mean = tf.reduce_mean(channel_intensity)
                ci_max = tf.reduce_max(channel_intensity)
                ci_min = tf.reduce_min(channel_intensity)
                ci_std = tf.math.reduce_std(channel_intensity)
                ci_non_zeros = tf.math.count_nonzero(channel_intensity, dtype=tf.int32)
                ci_non_zeros_r = ci_non_zeros / tf.math.reduce_prod(channel_intensity.shape)

                print("{:}, mean:{:.3f}, max:{:.3f}, min:{:.3f}, std:{:.4f}, nonz:{:.3f}"
                      .format(layer_fm.name, ci_mean, ci_max, ci_min, ci_std, ci_non_zeros_r))

                layer_idx += 1

        if save_imgs:
            fname = mode + '_' + str(img_idx+batch*100) + '.png'
            plt.savefig('./result_fig_fm_sc/' + fname)



# TODO: move to model.py
def collect_record_output(self):
    with tf.device('CPU:0'):
        for layer in self.model.layers_record:
            #print(layer)

            #print(layer.record_output)

            if not (layer.name in self.model.dict_stat_w.keys()):
                #self.dict_stat_w[layer.name] = layer.record_output.numpy()
                self.model.dict_stat_w[layer.name] = tf.Variable(layer.record_output,trainable=False,name='collect_record_output')
            else:
                self.model.dict_stat_w[layer.name] = tf.concat([self.model.dict_stat_w[layer.name],layer.record_output],0)
                #prev = self.dict_stat_w[layer.name]
                #new = layer.record_output.numpy()
                #self.dict_stat_w[layer.name] = np.concatenate((prev,new),axis=0)
                #self.dict_stat_w[layer.name].concate

    #print(self.dict_stat_w['conv1'].shape)
    #assert False, 'start here'

    #assert False



########################################
# (on_test_end)
########################################
def postproc(self,logs):

    #
    postproc_sel={
        'ANN': postproc_ann,
        'SNN': postproc_snn,
    }
    postproc_sel[self.model.nn_mode](self,logs)

def postproc_ann(self,logs):

    if self.conf.mode=='inference':
        #
        if self.conf.f_write_stat:
            lib_snn.weight_norm.save_act_stat(self)
            return

        if self.f_vth_set_and_norm:
            lib_snn.calibration.vth_set_and_norm(self)



        # TODO
        if (not self.conf.full_test) and conf._run_for_visual_debug:
        #dnn_snn_compare=True
        #if (not self.conf.full_test) and dnn_snn_compare:
            #idx=7
            plot_dnn_act(self)
            #plot_act_dist(self)
            #pass

        #
        if False:
            print('non zero ratio in ann act (postproc)')
            #for layer in self.layers_w_kernel:
            for layer in self.model.layers_w_kernel:

                if isinstance(layer, lib_snn.layers.Conv2D):
                    axis = [1, 2, 3]
                elif isinstance(layer, lib_snn.layers.Dense):
                    axis = [1]
                else:
                    assert False

                non_zero = tf.math.count_nonzero(layer.record_output, dtype=tf.float32, axis=axis)
                non_zero_r = non_zero / tf.cast(tf.reduce_prod(layer.record_output.shape[1:]),tf.float32)

                non_zero_r_m = tf.reduce_mean(non_zero_r)

                print(layer.name)
                print(non_zero_r_m)

        if False:
            print('plot nonzero ratio')
            figs, axes = plt.subplots(5, 5, figsize=(12,10))
            for layer in self.model.layers_w_kernel:
                axe = axes.flatten()[layer.depth]

                nonzero = self.nonzero_ratios[layer.name]
                axe.plot(nonzero)

                #(n, bins, patches) = axe.hist(act, bins=bins)
                axe.axhline(y=tf.reduce_mean(nonzero),color='r')
                #axe.set_ylim([0, n[10]])
                axe.set_title(layer.name)

            plt.show()


#
def plot_act_dist(self,fig=None):
    from lib_snn.sim import glb_plot

    if fig is None:
        fig = glb_plot

    #for layer in self.layers_w_kernel:
    for layer in self.model.layers_w_kernel:
        axe = fig.axes.flatten()[layer.depth]
        if self.model.nn_mode=='ann':
            act = layer.record_output.numpy().flatten()
            bins = np.arange(0,1,0.03)
        else:
            act = layer.act.spike_count_int.numpy().flatten()
            bins = np.arange(0,self.conf.time_step,1)

        (n, bins, patches) = axe.hist(act,bins=bins)
        axe.axvline(x=np.max(act), color='b')
        axe.set_ylim([0,n[10]])
        axe.set_title(layer.name)

    plt.show()

# TODO: move
def plot_spike_time_diff_hist_dnn_ann(self, fig=None):
    from lib_snn.sim import glb_plot

    diffs = collections.OrderedDict()

    if fig is None:
        fig = glb_plot

    #for layer in self.layers_w_kernel:
    for layer in self.model.layers_w_kernel:
        #if self.conf.snn_output_type == 'VMEM' and layer.name=='predictions':
        if self.conf.snn_output_type == 'VMEM' and layer == self.model.layers_w_kernel[-1]:
            continue

        axe = fig.axes.flatten()[layer.depth]
        ann_act = self.model_ann.get_layer(layer.name).record_output

        estimated_first_spike_time = self.conf.n_init_vth/ann_act
        first_spike_time = layer.act.first_spike_time
        diff = first_spike_time - estimated_first_spike_time
        diff = tf.where(tf.math.is_inf(diff),tf.constant(np.nan,shape=diff.shape),diff)
        diff = tf.where(tf.less(ann_act,tf.constant(1/self.conf.time_step)),tf.constant(np.nan,shape=diff.shape),diff)

        diff = tf.reshape(diff,shape=-1)
        diffs[layer.name] = diff

        #self.estimated_first_spike_time = estimated_first_spike_time
        #self.first_spike_time = first_spike_time
        #self.diff = diff

        #print(estimated_first_spike_time)
        #print(first_spike_time)
        #print(diff)

        axe.hist(diff)
        #axe.axvline(x=np.max(act), color='b')
        #axe.set_ylim([0, n[10]])
        axe.set_title(layer.name)

    plt.show()

    print('time diff')
    #for layer in self.layers_w_kernel:
    for layer in self.model.layers_w_kernel:
        if self.conf.snn_output_type == 'VMEM' and layer == self.model.layers_w_kernel[-1]:
            continue
        diff_mean=tf.experimental.numpy.nanmean(tf.abs(diffs[layer.name]))
        print(diff_mean)


    print('\n first spike time of previous layer - bias en time of layer')
    #for (idx_layer, layer) in enumerate(self.layers_w_kernel):
    for (idx_layer, layer) in enumerate(self.model.layers_w_kernel):
        if self.conf.snn_output_type == 'VMEM' and layer == self.model.layers_w_kernel[-1]:
            continue
        if idx_layer!=0:
            first_spike_time_in_mean = tf.experimental.numpy.nanmean(prev_layer.act.first_spike_time)
            bias_en_time = tf.reduce_mean(layer.bias_en_time)
            bias_en_time = tf.cast(bias_en_time,tf.float32)
            print(first_spike_time_in_mean - bias_en_time)
        prev_layer = layer

    #print('non zero ratio')
    print('\n act diff - dnn_act - snn_act')
    errs = []
    #for layer in self.layers_w_kernel:
    for layer in self.model.layers_w_kernel:
        if self.conf.snn_output_type == 'VMEM' and layer == self.model.layers_w_kernel[-1]:
            continue

        if self.conf.bias_control:
            time = tf.cast(self.conf.time_step - tf.reduce_mean(layer.bias_en_time), tf.float32)
        else:
            time = self.conf.time_step

        ann_act = self.model_ann.get_layer(layer.name).record_output
        snn_act = layer.act.spike_count_int/time

        err = tf.reduce_mean(tf.abs(ann_act-snn_act))
        errs.append(err)
        print('ann_act_mean - {}, err - {}'.format(tf.reduce_mean(ann_act),err))
    print('total mean - {}'.format(tf.reduce_mean(errs)))

#    print('\n act diff - dnn_act - snn_act (only non zero dnn_act)')
#    for layer in self.layers_w_kernel:
#        if self.conf.snn_output_type == 'VMEM' and layer == self.layers_w_kernel[-1]:
#            continue
#        ann_act = self.model_ann.get_layer(layer.name).record_output
#        ann_act = tf.where(ann_act==0,np.nan,ann_act)
#        snn_act = layer.act.spike_count_int/self.conf.time_step
#
#        #err = tf.reduce_mean(tf.abs(ann_act-snn_act))
#        err = tf.experimental.numpy.nanmean(tf.abs(ann_act-snn_act))
#
#        print('ann_act_mean - {}, err - {}'.format(tf.experimental.numpy.nanmean(ann_act),err))




# TODO: move
def plot_dnn_act(self, layers=None, idx=None):
    from lib_snn.sim import glb_plot

    #
    for idx_layer, layer in enumerate(glb_plot.layers):
        layer = self.model.get_layer(layer)
        axe = glb_plot.axes.flatten()[idx_layer]
        idx_neuron = glb_plot.idx_neurons[idx_layer]

        act = layer.record_output.numpy().flatten()[idx_neuron]
        axe.axhline(y=act, color='b')

        idx_bias = idx_neuron % layer.bias.shape[0]
        axe.axhline(y=layer.bias[idx_bias], color='m')

        #if self.conf.f_w_norm_data:
            #axe.axhline(y=(self.dict_stat_r[layer.name]/self.norm_b[layer.name]).flatten()[idx_neuron], color='r')

    if False:
        # layers in model
        idx=glb_plot.idx
        #for layer in self.layers_w_kernel:
        for layer in self.model.layers_w_kernel:
            #print(layer.name)
            #print(layer.depth)

            axe = glb_plot.axes.flatten()[layer.depth]

            # idx=0,0,0,0
            #spike = layer.act.get_spike_count_int().numpy().flatten()[idx]
            # print('{} - {}, {}'.format(glb_t.t,spike,spike/glb_t.t))
            #lib_snn.util.plot(glb_t.t, spike / glb_t.t, axe=axe)
            act = layer.record_output.numpy().flatten()[idx]
            axe.axhline(y=act, color='b')


def postproc_snn(self, logs):

    #
    # TODO: to be updated
    #postproc_snn_sel={
        #'inference': postproc_snn_infer,
        #'train': postproc_snn_train,
        #'load_and_train': postproc_snn_train,
    #}
    #postproc_snn_sel[self.model.run_mode](self)


    #if self.conf.train:
    if self.conf.mode=='train' or self.conf.mode=='load_and_train':
        postproc_snn_train(self)
    else:
        postproc_snn_infer(self)

        #
        #spike_count_epoch_end(self,0,logs)
        spike_count_epoch_end(self,self.epoch,logs,self.test_ds_num)

    #if conf.debug_grad: # only for debug
    if False:
        gradient_epoch_end(self,self.epoch)


def postproc_snn_infer(self):
    # results
    #cal_results(self)

    #
    if self.f_vth_set_and_norm:
        lib_snn.calibration.vth_set_and_norm(self)


    if self.calibration_bias and (not self.conf.calibration_bias_up_prog):
        lib_snn.calibration.calibration_bias_set(self)

    # print results
    #print_results(self)

    #
    save_condition = (self.conf.full_test and \
                     not (self.run_for_calibration_ML or self.run_for_vth_search or self.f_vth_set_and_norm))\
                     or self.conf.calibration_idx_test or self.conf.vth_search_idx_test
    save_condition = save_condition and self.f_save_result

    #if self.conf.full_test and not (self.run_for_calibration_ML or self.run_for_vth_search or self.f_vth_set_and_norm):
    if save_condition:
        save_results(self)


    if (not self.conf.full_test) and conf._run_for_visual_debug:
    #dnn_snn_compare = True
    #if dnn_snn_compare:
        dnn_snn_compare_func(self)



    if self.conf.verbose and self.conf.bias_control:
        print('bias en time')

        # for layer in self.layers_w_kernel:
        for layer in self.model.layers_w_kernel:
            print('{:<8} - {:3d}'.format(layer.name, tf.reduce_mean(layer.bias_en_time)))

#
def postproc_snn_train(self):
    pass


########################################
# (on_train_batch_end)
########################################
def postproc_batch_train(self):
#print('postproc_batch_train')
    pass

    #
    postproc_batch_train_sel={
        'ANN': postproc_batch_train_ann,
        'SNN': postproc_batch_train_snn,
    }
    postproc_batch_train_sel[self.model.nn_mode](self)



#
def postproc_batch_train_ann(self):
    pass


def postproc_batch_train_snn(self):
    pass
    spike_count_batch_end(self)


########################################
# (on_train_epoch_end)
########################################
def postproc_epoch_train(self,epoch,logs):

    #
    postproc_epoch_train_sel={
        'ANN': postproc_epoch_train_ann,
        'SNN': postproc_epoch_train_snn,
    }
    postproc_epoch_train_sel[self.model.nn_mode](self,epoch,logs)


def postproc_epoch_train_ann(self,epoch,logs):
    pass

def postproc_epoch_train_snn(self,epoch,logs):
    spike_count_epoch_end(self,epoch,logs,self.train_ds_num)


########################################
# spike count
########################################

def spike_count_epoch_init(self):
    #
    for neuron in self.model.layers_w_neuron:
        self.list_spike_count[neuron.name] = 0

    self.spike_count_total = 0

def spike_count_batch_end(self):
    #tf.summary.scalar('best_acc_val', data=self.best, step=epoch)
    #logs['best_acc_val'] = self.best

    #if not hasattr(self,'list_spike_count'):
    #    self.list_spike_count = collections.OrderedDict()

    #if not hasattr(self,'spike_count_total'):
    #    self.spike_count_total = 0

    #
    strategy = tf.distribute.get_strategy()
    if isinstance(strategy,tf.distribute.OneDeviceStrategy):
        for neuron in self.model.layers_w_neuron:
            name = neuron.name
            spike_count = tf.reduce_sum(neuron.act.spike_count_int)
            self.list_spike_count[name] += spike_count
            self.spike_count_total += spike_count
    else:
        for neuron in self.model.layers_w_neuron:
            name = neuron.name
            spike_count = tf.reduce_sum(neuron.act.spike_count_int.values)
            self.list_spike_count[name] += spike_count
            self.spike_count_total += spike_count


def spike_count_batch_end_one_device(self):
    for neuron in self.model.layers_w_neuron:
        name = neuron.name
        spike_count = tf.reduce_sum(neuron.act.spike_count_int)
        self.list_spike_count[name] += spike_count
        self.spike_count_total += spike_count


def spike_count_epoch_end(self,epoch,logs,num_ds):

    #training = K.learning_phase()  -> epoch end - training False
    #training = self.conf.mode=='train'
    #if training:
        #num_data = self.train_ds_num
    #else:
        #num_data = self.test_ds_num
    num_data = num_ds

    for neuron in self.model.layers_w_neuron:
        self.list_spike_count[neuron.name]/=num_data
    self.spike_count_total/=num_data

    #
    tf.summary.scalar('s_count', data=self.spike_count_total, step=epoch)
    logs['s_count'] = self.spike_count_total

    if 'best_val_acc' in logs.keys():
        if logs['val_acc'] == logs['best_val_acc']:
            self.spike_count_total_best = self.spike_count_total

        logs['best_s_count'] = self.spike_count_total_best

        with self.writer.as_default():
            tf.summary.scalar('bset_s_count', data=self.spike_count_total_best, step=epoch)
            self.writer.flush()


def gradient_epoch_end(self,epoch):
    #print('here')
    #for layer in self.model.layers:
    #    print(layer.name)


    #print('')
    #print('gradients')

    g_mean_list = []
    g_max_list = []
    g_min_list = []
    g_std_list = []

    for gv in self.model.grads_and_vars:
        g = gv[0]
        v = gv[1]
        name = v.name
        g_mean = tf.reduce_mean(g)
        g_max = tf.reduce_max(g)
        g_min = tf.reduce_min(g)
        g_std = tf.math.reduce_std(g)
        #if name == 'conv1/kernel:0':
        #    print("{:} - mean: {:e}, max: {:e}, min: {:e}, std: {:e}".format(name, g_mean, g_max, g_min, g_std))
        if False:
            print("{:} - mean: {:e}, max: {:e}, min: {:e}, std: {:e}".format(name, g_mean, g_max, g_min, g_std))

        #with self.writer.as_default():
        #    tf.summary.scalar('grad_mean_'+name, data=g_mean, step=epoch)
        #    self.writer.flush()

        g_mean_list.append(g_mean)
        g_max_list.append(g_max)
        g_min_list.append(g_min)
        g_std_list.append(g_std)


    #
    grad_mean_mean = tf.reduce_mean(g_mean_list)
    grad_mean_max = tf.reduce_mean(g_max_list)
    grad_mean_min = tf.reduce_mean(g_min_list)
    grad_mean_std = tf.reduce_mean(g_std_list)

    with self.writer.as_default():
        tf.summary.scalar('grad_mean', data=grad_mean_mean, step=epoch)
        tf.summary.scalar('grad_max', data=grad_mean_max, step=epoch)
        tf.summary.scalar('grad_min', data=grad_mean_min, step=epoch)
        tf.summary.scalar('grad_std', data=grad_mean_std, step=epoch)
        self.writer.flush()



#
def dnn_snn_compare_func(self):
    #dnn_snn_compare = True
    dnn_snn_compare = False

    #
    #plot_dnn_act(self)

    #
    if (not self.conf.full_test) and dnn_snn_compare:
        fig = lib_snn.sim.GLB_PLOT()
        plot_act_dist(self, fig)

    #
    if (not self.conf.full_test) and dnn_snn_compare and self.model.nn_mode=='SNN':
        # difference btw - estimated first spike time (DNN,vth/act) and first spike time (SNN)
        fig = lib_snn.sim.GLB_PLOT()
        plot_spike_time_diff_hist_dnn_ann(self,fig)

    #if (not self.conf.full_test) and self.conf.verbose_visual:
        #plt.show(block=False)
        #plt.show()
        #plt.draw()
        #plt.pause(0.01)



#
def cal_results(self):
    #assert False, 'check call it each epoch end - move to batch end routine'

    #if conf.snn_training_spatial_first:
    if True:
        self.results_acc = np.zeros(self.model.num_accuracy_time_point)
        self.results_spike_int = np.zeros(self.model.num_accuracy_time_point)
        self.results_spike = np.zeros(self.model.num_accuracy_time_point)
        self.results_loss = np.zeros(self.model.num_accuracy_time_point)

        for idx in range(self.model.num_accuracy_time_point):
            self.results_acc[idx] = self.model.accuracy_results[idx]['acc'].numpy()
            if 'loss' in self.model.accuracy_results[idx].keys():
                self.results_loss[idx] = self.model.accuracy_results[idx]['loss'].numpy()
            else:
                self.results_loss[idx] = np.NaN


        for layer_spike in self.model.total_spike_count_int.values():
            self.results_spike_int += layer_spike

        for layer_spike in self.model.total_spike_count.values():
            self.results_spike += layer_spike

        self.results_df = pd.DataFrame({'time step': self.model.accuracy_time_point, 'accuracy': self.results_acc,
                                        'spike count': self.results_spike / self.test_ds_num, 'loss': self.results_loss})
        self.results_df.set_index('time step', inplace=True)
    else:
        self.results_spike = 0
        self.results_spike_layer = collections.OrderedDict()
        for layer in self.model.layers:
            if isinstance(layer, lib_snn.activations.Activation):
                print(layer.name)
                act = layer.act
                if isinstance(act, lib_snn.neurons.Neuron):
                    self.results_spike += tf.reduce_sum(layer.act.spike_count)


def print_results(self):
    pd.set_option('display.float_format', '{:.4g}'.format)

    #
    # df=pd.DataFrame({'time step': model.accuracy_time_point, 'spike count': list(model.total_spike_count[:,-1]),'accuracy': accuracy_result})
    # df=pd.DataFrame({'time step': model.accuracy_time_point, 'accuracy': accuracy_result, 'spike count': model.total_spike_count_int[:,-1]})
    print(self.results_df)

def save_results(self):

    model_dataset = self.conf.model+'_'+self.conf.dataset

    # TODO: modify
    if self.conf.f_w_norm_data:
        #_norm = 'M999'
        _norm = self.conf.norm_stat
    else:
        _norm = 'NO'

    _n = self.conf.n_type
    _in = self.conf.input_spike_mode
    _nc = self.conf.neural_coding
    _ts = self.conf.time_step
    _tsi = self.conf.time_step_save_interval
    _vth = self.conf.n_init_vth

    config = 'norm-{}_n-{}_in-{}_nc-{}_ts-{}-{}_vth-{}'.format(_norm,_n,_in,_nc,_ts,_tsi,_vth)

    # vth search
    if self.conf.vth_search:
        config += '_vs'

    #
    if self.conf.vth_toggle:
        config += '_vth-tg-'+str(self.conf.vth_toggle_init)

    #
    if self.conf.calibration_weight:
        config += '_cal-w'

    #
    #if self.conf.calibration_weight_post:
    if self.conf.calibration_weight_act_based:
        config += '_cal-w-a'

    # calibration bias (ICLR-21)
    if self.conf.calibration_bias_ICLR_21:
        config += '_cal-b-LR21'

    # calibration bias (ICML-21)
    if self.conf.calibration_bias_ICML_21:
        config += '_cal-b-ML21'

    # calibration bias - new
    if self.conf.calibration_bias_new:
        config += '_cal-b-new'

    # calibration vmem (ICML-21)
    if self.conf.calibration_vmem_ICML_21:
        config += '_cal-v-ML21'

    # vth_search_test
    if self.conf.vth_search_idx_test:
        config += '_vth-test-idx-' + str(self.conf.vth_search_idx)

    # calibration test
    if self.conf.calibration_idx_test:
        config += '_cal-test-idx-' + str(self.conf.calibration_idx)

    # bias control
    if self.conf.bias_control:
        config += '_bc'


    # dynamic_bn test
    if self.conf.dynamic_bn_test:
        config += '_dbn-' + str(self.conf.dynamic_bn_dnn_act_scale) + '-' + str(self.conf.dynamic_bn_test_const)

    #
    file = config+'.xlsx'

    # set path
    path = self.conf.root_results
    path = os.path.join(path,conf.exp_set_name)
    path = os.path.join(path,model_dataset)
    f_name_result = os.path.join(path,file)

    #
    os.makedirs(path,exist_ok=True)

    #
    print('output file: {}'.format(f_name_result))
    self.results_df.to_excel(f_name_result)


########################################
# processing kernel
########################################
def bn_fusion(self):
    print('---- BN Fusion ----')
    # for integrated model v1
    ## bn fusion
    #if (self.model.nn_mode == 'ANN' and self.conf.f_fused_bn) or (self.model.nn_mode == 'SNN'):
    #    for layer in self.model.layers:
    #        if hasattr(layer, 'bn') and (layer.bn is not None):
    #            layer.bn_fusion()

    # v2
    # TODO: parameterize, model graph traverse
    #assert self.model.name
    # currently only for VGG16, CIFAR10

    if (self.model.nn_mode == 'ANN' and self.conf.f_fused_bn) or (self.model.nn_mode == 'SNN'):
        for layer in self.model.layers:
            if isinstance(layer,lib_snn.layers.BatchNormalization):
                l_name = layer.name.split('bn_')[-1]
                self.model.get_layer(l_name).bn_fusion_v2(layer)



    print('---- BN Fusion Done ----')

# TODO: move?
def w_norm_data(self):
    # weight normalization - data based

    print('---- Data-based weight normalization ----')

    self.dict_stat_r = collections.OrderedDict()

    self.norm = collections.OrderedDict()
    self.norm_b = collections.OrderedDict()

    #path_stat=self.conf.path_stat
    ##f_name_stat_pre=self.conf.prefix_stat
    #model_dataset = self.conf.model+'_'+self.conf.dataset
    #f_name_stat_pre=model_dataset
    #path_stat = os.path.join(path_stat,model_dataset)

    #path_stat = os.path.join(self.path_model_load,self.conf.path_stat)
    path_stat = os.path.join(config_glb.path_stat)

    #stat_conf=['max','mean','max_999','max_99','max_98']

    f_stat=collections.OrderedDict()
    r_stat=collections.OrderedDict()

    # choose one
    #stat='max'
    #stat='mean'
    #stat='max_999'
    #stat='max_998'
    #stat='max_997'
    #stat='max_99'
    #stat='max_98'
    #stat='max_95'
    #stat='max_90'
    #stat='max_80'
    #stat='max_75'
    #stat='max_25'
    #stat='max_60'
    #stat='max_40'
    #stat='median'
    #stat='mean'
    stat=self.conf.norm_stat

    #if self.conf.calibration_weight:?
        #stat='max_90'

    #for idx_l, l in enumerate(self.list_layer_name):
    #for idx_l, l in enumerate(self.list_layer):
    #for idx_l, l in enumerate(self.model.layers_w_act):
    for idx_l, l in enumerate(self.model.layers_w_neuron):
        if l.act.loc == 'IN':
            continue

        key=l.name+'_'+stat

        #f_name_stat = f_name_stat_pre+'_'+key
        f_name_stat = key
        f_name=os.path.join(path_stat,f_name_stat)
        f_stat[key]=open(f_name,'r')
        r_stat[key]=csv.reader(f_stat[key])

        # TODO: np -> tf.Variable
        for row in r_stat[key]:
            #self.dict_stat_r[l]=np.asarray(row,dtype=np.float32).reshape(self.list_shape[l][1:])
            self.dict_stat_r[l.name]=np.asarray(row,dtype=np.float32).reshape(l.output_shape_fixed_batch[1:])

        f_stat[key].close()

    #
    f_norm = np.max
    #f_norm = np.median
    #f_norm = np.mean
    #f_norm = lambda x: np.percentile(x,99.9)


    if stat=='max_999':
        f_norm = lambda x: np.percentile(x,99.9)

    #w_norm_data_layer_wise(self,f_norm)
    w_norm_data_channel_wise(self,f_norm,stat,dict_stat=self.dict_stat_r)




#
def w_norm_data_layer_wise(self, f_norm):

    print('layer-wise normalization')


    # for idx_l, l in enumerate(self.list_layer_name):
    #for idx_l, l in enumerate(self.list_layer):
    if 'VGG' in self.conf.model:
        #for idx_l, l in enumerate(self.layers_w_kernel):
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            stat = self.dict_stat_r[l.name]
            norm = f_norm(stat)
            #norm = np.where(norm==0, 1, norm)

            if idx_l == 0:
                self.norm[l.name] = norm
            else:
                self.norm[l.name] = norm / self.norm_b[prev_name]

            #self.norm_b[l.name] = f_norm(self.dict_stat_r[l.name])
            self.norm_b[l.name] = norm
            prev_name=l.name
    else:
        assert False


    # print
    #print('norm weight')
    #for k, v in self.norm.items():
        #print(k + ': ' + str(v))

    #print('norm bias')
    #for k, v in self.norm_b.items():
        #print(k + ': ' + str(v))

    #
    # for name_l in self.list_layer_name:
    #for layer in self.layers_w_kernel:
    for layer in self.model.layers_w_kernel:
        # layer = self.model.get_layer(name=name_l)
        layer.kernel = layer.kernel / self.norm[layer.name]
        layer.bias = layer.bias / self.norm_b[layer.name]


#
def w_norm_data_channel_wise(self, f_norm, stat, dict_stat=None):

    print('channel-wise normalization')

    #
    f_norm = lambda x: np.max(x, axis=0)

    # for idx_l, l in enumerate(self.list_layer_name):
    #for idx_l, l in enumerate(self.list_layer):
    if 'VGG' in self.conf.model:
        #for idx_l, l in enumerate(self.layers_w_kernel):
        for idx_l, l in enumerate(self.model.layers_w_kernel):
            if dict_stat is None:
                stat_r = lib_snn.calibration.read_stat(self,l,stat)
            else:
                stat_r = self.dict_stat_r[l.name]
            stat_r = stat_r.reshape(-1,stat_r.shape[-1])

            if isinstance(l,lib_snn.layers.InputGenLayer) or (l.last_layer and (self.conf.snn_output_type!='SPIKE')):
                norm = 1
            else:
                norm = f_norm(stat_r)
                norm = np.where(norm == 0, 1, norm)

            #if isinstance(l,lib_snn.layers.Dense):
                #norm = f_norm(norm)

            if idx_l == 0:
                #self.norm[l.name] = norm
                self.norm[l.name] = norm
            else:
                #stat = self.dict_stat_r[l.name]
                #stat = stat.reshape(-1, stat.shape[-1])
                #norm = f_norm(stat)

                #self.norm[l.name] = (f_norm(stat).T / self.norm[prev_name]).T
                self.norm[l.name] = norm / np.expand_dims(self.norm_b[prev_name],axis=0).T

                #print(self.norm[l.name])
                #assert False


            #self.norm_b[l.name] = f_norm(self.dict_stat_r[l.name])
            self.norm_b[l.name] = norm
            prev_name=l.name

        #
        # for visual debug
        #lib_snn.util.print_act_stat_r(self)
    elif 'ResNet' in self.conf.model:

#        # block_norm_in set
#        block_norm_in_name = collections.OrderedDict()
#        block_norm_out_name = collections.OrderedDict()
#        for idx_l, l in enumerate(self.model.layers_w_act):
#            if not ('conv' in l.name):
#                continue
#
#            #print(l.name)
#            conv_block_name = l.name.split('_')
#            conv_block_name = conv_block_name[0] + '_' + conv_block_name[1]
#
#            #print(conv_block_name)
#
#            if (idx_l==0) and (not 'block' in conv_block_name):
#                block_norm_in_name[conv_block_name] = None
#                block_norm_out_name[conv_block_name] = conv_block_name
#                next_block_norm_in_name = conv_block_name
#            else:
#                if not (conv_block_name in block_norm_in_name.keys()):
#                    block_norm_in_name[conv_block_name] = next_block_norm_in_name
#                    block_norm_out_name[conv_block_name] = conv_block_name+'_out'
#
#                    next_block_norm_in_name = block_norm_out_name[conv_block_name]
#

            #print(l.name)
        #    stat = self.dict_stat_r[l.name]

        #
        #for l in self.model.layers_w_kernel:
        #    if not ('conv' in l.name):
        #        continue
        #
        #    conv_block_name = l.name.split('_')
        #    conv_block_name = conv_block_name[0] + '_' + conv_block_name[1]
        #   #print('layer: {}, norm in: {}, norm out: {}'.format(l.name,block_norm_in_name[conv_block_name],block_norm_out_name[conv_block_name]))

        #assert False

        for idx_l, l in enumerate(self.model.layers_w_kernel):
            #print(l.name)

            #
            #if dict_stat is None:
                #stat_r = lib_snn.calibration.read_stat(self,l,stat)
            #else:
                #stat_r = self.dict_stat_r[l.name]
            #stat_r = stat_r.reshape(-1,stat_r.shape[-1])

            #if isinstance(l,lib_snn.layers.InputGenLayer):
                #norm = 1
            #else:
                #norm = f_norm(stat_r)
                ##norm = np.where(norm == 0, 1, norm)

            #
            if False:
            #if True:
                test_list = ['conv1_conv']
                if not l.name in test_list:
                    continue
                else:
                    stat_r = self.dict_stat_r[l.name]
                    stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                    norm = f_norm(stat_r)
                    norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)

                    norm_b = norm
                    norm_next = norm

                    #norm = tf.expand_dims(norm,axis=0)
                    #norm = tf.expand_dims(norm,axis=0)
                    #norm = tf.expand_dims(norm,axis=0)
                    self.norm[l.name] = norm
                    self.norm_b[l.name] = norm_b

                    if False:
                        norm_next_e = tf.expand_dims(norm_next,axis=0)
                        norm_next_e = tf.expand_dims(norm_next_e,axis=0)
                        norm_next_e = tf.expand_dims(norm_next_e,axis=3)
                        self.norm['conv2_block1_conv1'] = 1/norm_next_e
                        #self.norm_b['conv2_block1_conv1'] = 1/norm
                        self.norm['conv2_block1_conv0'] = 1/norm_next_e
                        #self.norm_b['conv2_block1_conv0'] = 1/norm

                        #stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        #norm = f_norm(stat_r)
                        #norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        #self.norm['conv2_block1_conv1'] = norm / np.expand_dims(norm_next, axis=0).T
                        #self.norm_b['conv2_block1_conv1'] = norm

                        stat_r = self.dict_stat_r['conv2_block1_conv1']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        self.norm['conv2_block1_conv1'] = norm / np.expand_dims(norm_next, axis=0).T
                        self.norm_b['conv2_block1_conv1'] = norm

                        norm_next = norm
                        #norm_next_e = tf.expand_dims(norm_next,axis=0)
                        #norm_next_e = tf.expand_dims(norm_next_e,axis=0)
                        #norm_next_e = tf.expand_dims(norm_next_e,axis=3)
                        stat_r = self.dict_stat_r['conv2_block1_out']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        self.norm['conv2_block1_conv2'] = norm / np.expand_dims(norm_next, axis=0).T
                        self.norm_b['conv2_block1_conv2'] = norm
                        #self.norm['conv2_block1_conv2'] = 1/norm_next_e


                    if False:
                    #if True:
                        stat_r = self.dict_stat_r['conv2_block1_out']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        self.norm['conv2_block1_conv0'] = norm / np.expand_dims(norm_next, axis=0).T
                        self.norm_b['conv2_block1_conv0'] = norm

                        stat_r = self.dict_stat_r['conv2_block1_conv1']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        self.norm['conv2_block1_conv1'] = norm / np.expand_dims(norm_next, axis=0).T
                        self.norm_b['conv2_block1_conv1'] = norm

                        stat_r = self.dict_stat_r['conv2_block1_out']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        norm_next = self.norm_b['conv2_block1_conv1']
                        self.norm['conv2_block1_conv2'] = norm / np.expand_dims(norm_next, axis=0).T
                        self.norm_b['conv2_block1_conv2'] = norm

                        norm_next = norm
                        #self.norm['conv2_block2_conv0_i'] = 1/norm_next
                        #norm_next = tf.expand_dims(norm_next,axis=0)
                        #norm_next = tf.expand_dims(norm_next,axis=0)
                        #norm_next = tf.expand_dims(norm_next,axis=3)
                        #self.norm['conv2_block2_conv1'] = 1/norm_next


                        stat_r = self.dict_stat_r['conv2_block2_out']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        self.norm['conv2_block2_conv0_i'] = norm / norm_next
                        self.norm_b['conv2_block2_conv0_i'] = norm

                        stat_r = self.dict_stat_r['conv2_block2_conv1']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        self.norm['conv2_block2_conv1'] = norm / np.expand_dims(norm_next, axis=0).T
                        self.norm_b['conv2_block2_conv1'] = norm

                        stat_r = self.dict_stat_r['conv2_block2_out']
                        stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                        norm = f_norm(stat_r)
                        norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                        norm_next = self.norm_b['conv2_block2_conv1']
                        self.norm['conv2_block2_conv2'] = norm / np.expand_dims(norm_next, axis=0).T
                        self.norm_b['conv2_block2_conv2'] = norm

                        norm_next = norm
                        self.norm['conv2_block3_conv0_i'] = 1/norm_next
                        norm_next = tf.expand_dims(norm_next,axis=0)
                        norm_next = tf.expand_dims(norm_next,axis=0)
                        norm_next = tf.expand_dims(norm_next,axis=3)
                        self.norm['conv2_block3_conv1'] = 1/norm_next

                    continue

            #if not l.name in ['conv1_conv', 'conv2_block1_conv1', 'conv2_block1_conv0', 'conv2_block1_conv2']:
            #    continue

            if (idx_l==0):
                stat_r = self.dict_stat_r[l.name]
                stat_r = stat_r.reshape(-1, stat_r.shape[-1])
                norm = f_norm(stat_r)
                norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)
                self.norm[l.name] = norm



            elif (not ('conv' in l.name)) :
                #print('not conv - {}'.format(l.name))
                #print(prev_name)
                stat_r = self.dict_stat_r[l.name]
                stat_r = stat_r.reshape(-1,stat_r.shape[-1])
                norm = f_norm(stat_r)
                norm = tf.where(norm == 0.0, tf.ones(norm.shape), norm)

                #print(norm)
                self.norm[l.name] = norm / np.expand_dims(self.norm_b[prev_name],axis=0).T

                #if self.conf.verbose:
                    #print('l_name: - {}'.format(l.name))
                    #print('l_name_prev: - {}'.format(prev_name))
                    #print('')

            elif ('conv' in l.name):
                conv_block_name = l.name.split('_')
                conv_name = conv_block_name[2]
                conv_block_name = conv_block_name[0] + '_' + conv_block_name[1]

                #print('conv_name - {}'.format(conv_name))

                if 'conv0' in conv_name:
                    norm_l_name = self.model.block_norm_out_name[conv_block_name]
                    norm_prev_l_name = self.model.block_norm_in_name[conv_block_name]
                    #norm_prev_l_name = norm_l_name
                elif 'conv1' in conv_name:
                    norm_l_name = l.name
                    norm_prev_l_name = self.model.block_norm_in_name[conv_block_name]
                elif 'conv2' in conv_name:
                    norm_l_name = self.model.block_norm_out_name[conv_block_name]
                    norm_prev_l_name = conv_block_name+'_conv1'
                else:
                    assert False

                stat_r = self.dict_stat_r[norm_l_name]
                stat_r = stat_r.reshape(-1,stat_r.shape[-1])
                norm = f_norm(stat_r)
                norm = tf.where(norm==0.0, tf.ones(norm.shape), norm)

                stat_r_prev = self.dict_stat_r[norm_prev_l_name]
                stat_r_prev = stat_r_prev.reshape(-1, stat_r_prev.shape[-1])
                norm_prev = f_norm(stat_r_prev)
                norm_prev = tf.where(norm_prev==0.0, tf.ones(norm_prev.shape), norm_prev)

                #print('layer: {}, norm: {}, norm_prev: {}'.format(l.name,norm_l_name,norm_prev_l_name))

                if isinstance(l,lib_snn.layers.Identity):
                    self.norm[l.name] = norm / norm_prev
                else:
                    self.norm[l.name] = norm / np.expand_dims(norm_prev, axis=0).T

                #
                #print('norm_l_name: - {}'.format(norm_l_name))
                #print('norm_prev_l_name: - {}'.format(norm_prev_l_name))
                #print('')

            else:
                assert False

            self.norm_b[l.name] = norm
            prev_name=l.name

    else:
        assert False

    #assert False


    # print
    #print('norm weight')
    #for k, v in self.norm.items():
        #print(k + ': ' + str(v))

    #print('norm bias')
    #for k, v in self.norm_b.items():
        #print(k + ': ' + str(v))


    if False:
    #if True:
        #for layer in test_list:
        for layer_name in self.norm.keys():
            layer = self.model.get_layer(name=layer_name)
            layer.kernel = layer.kernel / self.norm[layer_name]

        for layer_name in self.norm_b.keys():
            layer = self.model.get_layer(name=layer_name)
            layer.bias = layer.bias / self.norm_b[layer_name]

    if True:
    #if False:
        #
        # for name_l in self.list_layer_name:
        #for layer in self.layers_w_kernel:
        for layer in self.model.layers_w_kernel:
            layer.kernel = layer.kernel / self.norm[layer.name]
            layer.bias = layer.bias / self.norm_b[layer.name]


    #print(self.model.get_layer('fc2').kernel)
    #assert False

    print('normalization - done')


def cal_total_num_neurons(self):
    total_num_neurons = 0
    for l in self.model.layers:
        if hasattr(l, 'act'):
            if isinstance(l.act, lib_snn.neurons.Neuron):
                total_num_neurons += l.act.num_neurons

    self.total_num_neurons = total_num_neurons


def set_leak_const(self):
    for idx_l, l in enumerate(self.model.layers_w_neuron):

        if isinstance(l, lib_snn.layers.InputGenLayer):
            continue

        stat_max= lib_snn.calibration.read_stat(self, l, 'max')
        stat_mean = lib_snn.calibration.read_stat(self, l, 'mean')

        print("{} - max: {}, mean: {}".format(l.name,stat_max,stat_mean))

    assert False


def print_total_num_neurons(self):
    print('total num neurons: {:}'.format(self.total_num_neurons))
