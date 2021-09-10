
import os
import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils

from tensorflow.python.eager import imperative_grad
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

#
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

#
import matplotlib.pyplot as plt

import numpy as np


##############################################################
# keras model flops
##############################################################
def get_flops(model,input_shape_one_batch):
    if False:
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1,*inputs.shape[1:]]) for inputs in model.inputs])
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
            return flops.total_float_ops

    forward_pass = tf.function(model.call,
                               input_signature=[tf.TensorSpec(shape=(1,)+input_shape_one_batch)])
    graph_info = profile(forward_pass.get_concrete_function().graph,
                         options=ProfileOptionBuilder.float_operation())

    flops = graph_info.total_float_ops
    return flops


##############################################################
# return latest saved model
##############################################################
def get_latest_saved_model(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    list_dir_sorted = list(sorted(os.listdir(path), key=mtime))

    latest_model = list_dir_sorted[-1]

    return latest_model


##############################################################
# output shape - Conv2D
# from tensorflow/.../convolutional.py
##############################################################
def cal_output_shape_Conv2D(data_format,input_shape,filters,kernel_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    kernel_size = utils.normalize_tuple(kernel_size,2,'kernel_size')
    strides = utils.normalize_tuple(strides,2,'strides')
    dilation_rate = utils.normalize_tuple(1,2,'dilation_rate')

    if data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                #padding='valid',
                padding='same',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [filters]).as_list()
    else:
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                #padding='valid',
                padding='same',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0], filters] + new_space).as_list()


##############################################################
# output shape - Conv2D
##############################################################
def cal_output_shape_Conv2D_pad_val(data_format,input_shape,filters,kernel_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    kernel_size = utils.normalize_tuple(kernel_size,2,'kernel_size')
    strides = utils.normalize_tuple(strides,2,'strides')
    dilation_rate = utils.normalize_tuple(1,2,'dilation_rate')

    if data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='valid',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space + [filters]).as_list()
    else:
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='valid',
                stride=strides[i],
                dilation=dilation_rate[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0], filters] + new_space).as_list()

##############################################################
# output shape - Pooling2D
##############################################################
def cal_output_shape_Pooling2D(data_format,input_shape,pool_size,strides):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()

    pool_size = utils.normalize_tuple(pool_size, 2, 'pool_size')
    strides = utils.normalize_tuple(strides, 2, 'strides')
    padding = 'same'

    if data_format == 'channels_first':
        rows = input_shape[2]
        cols = input_shape[3]
    else:
        rows = input_shape[1]
        cols = input_shape[2]

    rows = utils.conv_output_length(
            rows,
            pool_size[0],
            padding,
            strides[0]
        )

    cols = utils.conv_output_length(
            cols,
            pool_size[1],
            padding,
            strides[1]
        )

    if data_format == 'channels_first':
        return tensor_shape.TensorShape([input_shape[0], input_shape[1], rows, cols]).as_list()
    else:
        print(input_shape)
        return tensor_shape.TensorShape([input_shape[0], rows, cols, input_shape[3]]).as_list()



############################################################
#
############################################################
def post_proc_dummy_run(self):
    if self.f_1st_iter and self.conf.nn_mode=='ANN':
    #if f_dummy_run and self.conf.nn_mode=='ANN':
        print('1st iter - dummy run')
        self.f_1st_iter = False
        self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)


############################################################
# BN layer check
############################################################
def is_bn(l_name, layer):
    check_l_name = 'bn' in l_name
    check_layer_type = isinstance(layer,tf.keras.layers.BatchNormalization)

    if check_l_name or check_layer_type:
        return True
    else:
        return False


############################################################
## ground_truth_in_spike_time
## one-hot label -> spike time (first spike time, TTFS)
############################################################
def ground_truth_in_spike_time(one_hot_label,target_time,non_target_time):
    ground_truth_target=target_time
    ground_truth_non_target=non_target_time

    gt_spike_time = tf.where(
                        tf.equal(one_hot_label,0.0),\
                        tf.constant(ground_truth_non_target,dtype=tf.float32,shape=one_hot_label.shape), \
                        tf.constant(ground_truth_target,dtype=tf.float32,shape=one_hot_label.shape))

    return gt_spike_time


############################################################
##
############################################################
def dataset_shape_one_sample(dataset):
    for (images, labels_one_hot) in dataset:
        ret = images.shape[1:]
        break

    return ret


############################################################
## recording dnn activation
############################################################
def recording_dnn_act(self,inputs):

    extractor = tf.keras.Model(inputs=self.model.inputs, \
                           outputs=[layer.output for layer in self.model.layers if layer.name in self.list_layer_name_write_stat])

    layers_output = extractor(inputs)

    for layer_idx, layer_name in enumerate(self.list_layer_name_write_stat):
        self.dnn_act_list[layer_name] = np.asarray(layers_output[layer_idx])


############################################################
## collect dnn activation
############################################################
def collect_dnn_act(self,inputs):

    recording_dnn_act(self,inputs)

    for l_name in self.list_layer_name_write_stat:
        dict_stat_w = self.dict_stat_w[l_name]

        if self.f_1st_iter_stat:
            dict_stat_w.assign(self.dnn_act_list[l_name])
        else:
            self.dict_stat_w[l_name]=tf.concat([dict_stat_w,self.dnn_act_list[l_name]], 0)

    if self.f_1st_iter_stat:
        self.f_1st_iter_stat = False

############################################################
## plot
############################################################
def plot(self, x, y, mark):
    #plt.ion()
    #plt.hist(self.list_neuron['fc3'].vmem)
    plt.plot(x, y, mark)
    plt.draw()
    plt.pause(0.00000001)
    #plt.ioff()


def scatter(self, x, y, color, axe=None, marker='o'):
    if axe==None:
        plt.scatter(x, y, c=color, s=1, marker=marker)
        plt.draw()
        plt.pause(0.0000000000000001)
    else:
        axe.scatter(x, y, c=color, s=1, marker=marker)
        plt.draw()
        plt.pause(0.0000000000000001)

def figure_hold(self):
    plt.close("dummy")
    plt.show()


############################################################
## raster plot (animation)
############################################################
#@tfplot.autowrap
def debug_visual_raster(self,t):

    subplot_x, subplot_y = 4, 4

    num_subplot = subplot_x*subplot_y
    idx_print_s, idx_print_e = 0, 100

    #
    synchronized = False

    if t==0:
        plt_idx=0
        #plt.figure()
        plt.close("dummy")
        _, self.debug_visual_axes = plt.subplots(subplot_y,subplot_x)

        #for neuron_name, neuron in self.list_neuron.items():
        for layers in self.list_layer:
            neuron = layers.act


            #if not ('fc3' in neuron_name):
            #self.debug_visual_list_neuron[layer.name]=neuron

            axe = self.debug_visual_axes.flatten()[plt_idx]

            axe.set_ylim([0,neuron.out.numpy().flatten()[idx_print_s:idx_print_e].size])
            axe.set_xlim([0,self.ts])

            axe.grid(True)

            if synchronized:
                if(plt_idx>0):
                    axe.axvline(x=(plt_idx-1)*self.conf.time_fire_start, color='b')                 # integration
                axe.axvline(x=plt_idx*self.conf.time_fire_start, color='b')                         # fire start
                axe.axvline(x=plt_idx*self.conf.time_fire_start+self.conf.time_fire_duration, color='b') # fire end

            plt_idx+=1

    else:
        plt_idx=0
        #for neuron_name, neuron in self.debug_visual_list_neuron.items():
        for layers in self.list_layer:
            neuron = layers.act


            if synchronized:
                # synchronized - layer dependent
                t_fire_s = plt_idx*self.conf.time_fire_start
                t_fire_e = t_fire_s + self.conf.time_fire_duration
            else:
                # asynchronized
                t_fire_s = 0
                t_fire_e = self.conf.time_window

            if t >= t_fire_s and t < t_fire_e :
                if tf.reduce_sum(neuron.out) != 0.0:
                    idx_fire=tf.where(tf.not_equal(tf.reshape(neuron.out,[-1])[idx_print_s:idx_print_e],tf.constant(0,dtype=tf.float32)))

                #if tf.size(idx_fire) != 0:
                    axe = self.debug_visual_axes.flatten()[plt_idx]
                    self.scatter(tf.fill(idx_fire.shape,t),idx_fire,'r', axe=axe)

            plt_idx+=1

    if t==self.ts-1:
        plt.figure("dummy")



############################################################
# first_spike_time visualization
############################################################
def visual_first_spike_time(self):
    print('first spike time')
    _, axes = plt.subplots(4,4)
    idx_plot=0
    #for n_name, n in self.list_neuron.items():
    for layer in self.list_layer:
        n = layer.act

        #positive = n.first_spike_time > 0
        #print(n_name+'] min: '+str(tf.reduce_min(n.first_spike_time[positive]))+', mean: '+str(tf.reduce_mean(n.first_spike_time[positive])))
        #print(tf.reduce_min(n.first_spike_time[positive]))

        #positive=n.first_spike_time.numpy().flatten() > 0
        positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0)

        if not tf.equal(tf.size(positive),0):

            #min=np.min(n.first_spike_time.numpy().flatten()[positive,])
            #print(positive.shape)
            #min=np.min(n.first_spike_time.numpy().flatten()[positive])
            min=tf.reduce_min(positive)
            #mean=np.mean(n.first_spike_time.numpy().flatten()[positive,])

            #if self.conf.f_tc_based:
            #    fire_s=idx_plot*self.conf.time_fire_start
            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration
            #else:
            #    fire_s=idx_plot*self.conf.time_fire_start
            #    fire_e=idx_plot*self.conf.time_fire_start+self.conf.time_fire_duration

            #fire_s = n.time_start_fire
            #fire_e = n.time_end_fire

            fire_s = idx_plot * self.conf.time_fire_start
            fire_e = idx_plot * self.conf.time_fire_start + self.conf.time_fire_duration

            axe=axes.flatten()[idx_plot]
            #axe.hist(n.first_spike_time.numpy().flatten()[positive],bins=range(fire_s,fire_e,1))
            axe.hist(positive.numpy().flatten(),bins=range(fire_s,fire_e,1))

            axe.axvline(x=min.numpy(),color='b', linestyle='dashed')

            axe.axvline(x=fire_s)
            axe.axvline(x=fire_e)


        idx_plot+=1

##############################################################
# NOT VERIFIED
##############################################################

#self.cmap=matplotlib.cm.get_cmap('viridis')


# from tensorflow/python/eager/backprop.py
def gradient(self, target, sources, output_gradients=None):
    """Computes the gradient using operations recorded in context of this tape.

    Args:
      target: Tensor (or list of tensors) to be differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      output_gradients: a list of gradients, one for each element of
        target. Defaults to None.

    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None),
      one for each element in `sources`. Returned structure is the same as
      the structure of `sources`.

    Raises:
      RuntimeError: if called inside the context of the tape, or if called more
       than once on a non-persistent tape.
    """
    if self._tape is None:
        raise RuntimeError("GradientTape.gradient can only be called once on "
                           "non-persistent tapes.")
    if self._recording:
        if not self._persistent:
            self._pop_tape()
        else:
            logging.log_first_n(logging.WARN,
                                "Calling GradientTape.gradient on a persistent "
                                "tape inside it's context is significantly less "
                                "efficient than calling it outside the context (it "
                                "causes the gradient ops to be recorded on the "
                                "tape, leading to increased CPU and memory usage). "
                                "Only call GradientTape.gradient inside the "
                                "context if you actually want to trace the "
                                "gradient in order to compute higher order "
                                "derrivatives.", 1)

    flat_sources = nest.flatten(sources)
    flat_sources = [_handle_or_self(x) for x in flat_sources]

    if output_gradients is not None:
        output_gradients = [None if x is None else ops.convert_to_tensor(x)
                            for x in nest.flatten(output_gradients)]

    flat_grad = imperative_grad.imperative_grad(
        _default_vspace, self._tape, nest.flatten(target), flat_sources,
        output_gradients=output_gradients)

    if not self._persistent:
        self._tape = None

    grad = nest.pack_sequence_as(sources, flat_grad)
    return grad


def debug_visual(self, synapse, neuron, synapse_1, neuron_1, synapse_2, neuron_2, t):

    idx_print_s, idx_print_e = 0, 200

    ax=plt.subplot(3,4,1)
    plt.title('w_sum_in (max)')
    self.plot(t,tf.reduce_max(tf.reshape(synapse,[-1])).numpy(), 'bo')
    #plt.subplot2grid((2,4),(0,1),sharex=ax)
    plt.subplot(3,4,2,sharex=ax)
    plt.title('vmem (max)')
    #self.plot(t,tf.reduce_max(neuron.vmem).numpy(), 'bo')
    self.plot(t,tf.reduce_max(tf.reshape(neuron.vmem,[-1])).numpy(), 'bo')
    self.plot(t,tf.reduce_max(tf.reshape(neuron.vth,[-1])).numpy(), 'ro')
    #self.plot(t,neuron.out.numpy()[neuron.out.numpy()>0].sum(), 'bo')
    #plt.subplot2grid((2,4),(0,2))
    plt.subplot(3,4,3,sharex=ax)
    plt.title('# spikes (total)')
    #spike_rate=neuron.get_spike_count()/t
    self.plot(t,tf.reduce_sum(tf.reshape(neuron.spike_counter_int,[-1])).numpy(), 'bo')
    #self.plot(t,neuron.vmem.numpy()[neuron.vmem.numpy()>0].sum(), 'bo')
    #self.plot(t,tf.reduce_max(spike_rate), 'bo')
    #plt.subplot2grid((2,4),(0,3))
    plt.subplot(3,4,4,sharex=ax)
    plt.title('spike neuron idx')
    plt.grid(True)
    #plt.ylim([0,512])
    plt.ylim([0,neuron.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
    #plt.ylim([0,int(self.list_neuron['fc2'].dim[1])])
    plt.xlim([0,self.ts])
    #self.plot(t,np.where(self.list_neuron['fc2'].out.numpy()==1),'bo')
    #if np.where(self.list_neuron['fc2'].out.numpy()==1).size == 0:
    idx_fire=np.where(neuron.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
    if not len(idx_fire)==0:
        #print(np.shape(idx_fire))
        #print(idx_fire)
        #print(np.full(np.shape(idx_fire),t))
        self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')
        #self.scatter(t,np.argmax(neuron.get_spike_count().numpy().flatten()),'b')

    addr=0,0,0,6
    ax=plt.subplot(3,4,5)
    #self.plot(t,tf.reduce_max(tf.reshape(synapse_1,[-1])).numpy(), 'bo')
    self.plot(t,synapse_1[addr].numpy(), 'bo')
    plt.subplot(3,4,6,sharex=ax)
    #self.plot(t,tf.reduce_max(tf.reshape(neuron_1.vmem,[-1])).numpy(), 'bo')
    self.plot(t,neuron_1.vmem.numpy()[addr], 'bo')
    self.plot(t,neuron_1.vth.numpy()[addr], 'ro')
    plt.subplot(3,4,7,sharex=ax)
    #self.plot(t,neuron_1.vmem.numpy()[neuron_1.vmem.numpy()>0].sum(), 'bo')
    self.plot(t,neuron_1.spike_counter_int.numpy()[addr], 'bo')
    plt.subplot(3,4,8,sharex=ax)
    plt.grid(True)
    plt.ylim([0,neuron_1.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
    plt.xlim([0,self.ts])
    idx_fire=np.where(neuron_1.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
    #idx_fire=neuron_1.f_fire.numpy().flatten()[idx_print_s:idx_print_e]
    #idx_fire=neuron_1.f_fire.numpy()[0,0,0,0:10]
    #print(neuron_1.vmem.numpy()[0,0,0,1])
    #print(neuron_1.f_fire.numpy()[0,0,0,1])
    #print(idx_fire)
    if not len(idx_fire)==0:
        self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')

    addr=0,0,0,6
    ax=plt.subplot(3,4,9)
    #self.plot(t,tf.reduce_max(tf.reshape(synapse_2,[-1])).numpy(), 'bo')
    self.plot(t,synapse_2[addr].numpy(), 'bo')
    plt.subplot(3,4,10,sharex=ax)
    #self.plot(t,tf.reduce_max(tf.reshape(neuron_2.vmem,[-1])).numpy(), 'bo')
    self.plot(t,neuron_2.vmem.numpy()[addr], 'bo')
    self.plot(t,neuron_2.vth.numpy()[addr], 'ro')
    plt.subplot(3,4,11,sharex=ax)
    #self.plot(t,neuron_2.vmem.numpy()[neuron_2.vmem.numpy()>0].sum(), 'bo')
    self.plot(t,neuron_2.spike_counter_int.numpy()[addr], 'bo')
    plt.subplot(3,4,12,sharex=ax)
    plt.grid(True)
    plt.ylim([0,neuron_2.vmem.numpy().flatten()[idx_print_s:idx_print_e].size])
    plt.xlim([0,self.ts])
    idx_fire=np.where(neuron_2.out.numpy().flatten()[idx_print_s:idx_print_e]!=0.0)
    #idx_fire=neuron_2.f_fire.numpy().flatten()[idx_print_s:idx_print_e]
    #idx_fire=neuron_2.f_fire.numpy()[0,0,0,0:10]
    #print(neuron_2.vmem.numpy()[0,0,0,1])
    #print(neuron_2.f_fire.numpy()[0,0,0,1])
    #print(idx_fire)
    if not len(idx_fire)==0:
        self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')



    if t==self.ts-1:
        plt.figure()
        #plt.show()


#
def print_model_visual(self):
    print('print model visual')

    plt.subplot(631)
    h_conv1=plt.hist(self.list_layer['conv1'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv1'].bias.numpy(),0,h_conv1[0].max())
    plt.subplot(632)
    h_conv1_1=plt.hist(self.list_layer['conv1_1'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv1_1'].bias.numpy(),0,h_conv1_1[0].max())

    plt.subplot(634)
    h_conv2=plt.hist(self.list_layer['conv2'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv2'].bias.numpy(),0,h_conv2[0].max())
    plt.subplot(635)
    h_conv2_1=plt.hist(self.list_layer['conv2_1'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv2_1'].bias.numpy(),0,h_conv2_1[0].max())

    plt.subplot(637)
    h_conv3=plt.hist(self.list_layer['conv3'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv3'].bias.numpy(),0,h_conv3[0].max())
    plt.subplot(638)
    h_conv3_1=plt.hist(self.list_layer['conv3_1'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv3_1'].bias.numpy(),0,h_conv3_1[0].max())
    plt.subplot(639)
    h_conv3_2=plt.hist(self.list_layer['conv3_2'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv3_2'].bias.numpy(),0,h_conv3_2[0].max())

    plt.subplot2grid((6,3),(3,0))
    h_conv4=plt.hist(self.list_layer['conv4'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv4'].bias.numpy(),0,h_conv4[0].max())
    plt.subplot2grid((6,3),(3,1))
    h_conv4_1=plt.hist(self.list_layer['conv4_1'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv4_1'].bias.numpy(),0,h_conv4_1[0].max())
    plt.subplot2grid((6,3),(3,2))
    h_conv4_2=plt.hist(self.list_layer['conv4_2'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv4_2'].bias.numpy(),0,h_conv4_2[0].max())

    plt.subplot2grid((6,3),(4,0))
    h_conv5=plt.hist(self.list_layer['conv5'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv5'].bias.numpy(),0,h_conv5[0].max())
    plt.subplot2grid((6,3),(4,1))
    h_conv5_1=plt.hist(self.list_layer['conv5_1'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv5_1'].bias.numpy(),0,h_conv5_1[0].max())
    plt.subplot2grid((6,3),(4,2))
    h_conv5_2=plt.hist(self.list_layer['conv5_2'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['conv5_2'].bias.numpy(),0,h_conv5_2[0].max())

    plt.subplot2grid((6,3),(5,0))
    h_fc1=plt.hist(self.list_layer['fc1'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['fc1'].bias.numpy(),0,h_fc1[0].max())
    plt.subplot2grid((6,3),(5,1))
    h_fc2=plt.hist(self.list_layer['fc2'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['fc2'].bias.numpy(),0,h_fc2[0].max())
    plt.subplot2grid((6,3),(5,2))
    h_fc3=plt.hist(self.list_layer['fc3'].kernel.numpy().flatten())
    plt.vlines(self.list_layer['fc3'].bias.numpy(),0,h_fc3[0].max())

    #print(self.list_layer['fc2_bn'].beta.numpy())
    #print(self.list_layer['fc2_bn'].gamma)
    #print(self.list_layer['fc2_bn'].beta)

    plt.show()


def visual(self, t):
    #plt.subplot2grid((2,4),(0,0))
    ax=plt.subplot(2,4,1)
    plt.title('w_sum_in (max)')
    self.plot(t,tf.reduce_max(s_fc2).numpy(), 'bo')
    #plt.subplot2grid((2,4),(0,1),sharex=ax)
    plt.subplot(2,4,2,sharex=ax)
    plt.title('vmem (max)')
    self.plot(t,tf.reduce_max(self.list_neuron['fc2'].vmem).numpy(), 'bo')
    #plt.subplot2grid((2,4),(0,2))
    plt.subplot(2,4,3,sharex=ax)
    plt.title('# spikes (max)')
    self.plot(t,tf.reduce_max(self.list_neuron['fc2'].get_spike_count()).numpy(), 'bo')
    #self.scatter(np.full(np.shape),tf.reduce_max(self.list_neuron['fc2'].get_spike_count()).numpy(), 'bo')
    #plt.subplot2grid((2,4),(0,3))
    plt.subplot(2,4,4,sharex=ax)
    plt.title('spike neuron idx')
    plt.grid(True)
    plt.ylim([0,512])
    #plt.ylim([0,int(self.list_neuron['fc2'].dim[1])])
    plt.xlim([0,tw])
    #self.plot(t,np.where(self.list_neuron['fc2'].out.numpy()==1),'bo')
    #if np.where(self.list_neuron['fc2'].out.numpy()==1).size == 0:
    idx_fire=np.where(self.list_neuron['fc2'].out.numpy()==1)[1]
    if not len(idx_fire)==0:
        #print(np.shape(idx_fire))
        #print(idx_fire)
        #print(np.full(np.shape(idx_fire),t))
        self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')


    #plt.subplot2grid((2,4),(1,0))
    plt.subplot(2,4,5,sharex=ax)
    self.plot(t,tf.reduce_max(s_fc3).numpy(), 'bo')
    #plt.subplot2grid((2,4),(1,1))
    plt.subplot(2,4,6,sharex=ax)
    self.plot(t,tf.reduce_max(self.list_neuron['fc3'].vmem).numpy(), 'bo')
    #plt.subplot2grid((2,4),(1,2))
    plt.subplot(2,4,7,sharex=ax)
    self.plot(t,tf.reduce_max(self.list_neuron['fc3'].get_spike_count()).numpy(), 'bo')
    plt.subplot(2,4,8)
    plt.grid(True)
    #plt.ylim([0,self.list_neuron['fc3'].dim[1]])
    plt.ylim([0,self.num_class])
    plt.xlim([0,tw])
    idx_fire=np.where(self.list_neuron['fc3'].out.numpy()==1)[1]
    if not len(idx_fire)==0:
        self.scatter(np.full(np.shape(idx_fire),t,dtype=int),idx_fire,'r')






#
def print_act_after_w_norm(self):
    self.print_act_stat_r()

#
def print_act_stat_r(self):
    print('print activation stat')

    fig, axs = plt.subplots(6,3)

    axs=axs.ravel()

    #for idx_l, (name_l,stat_l) in enumerate(self.dict_stat_r):
    for idx, (key, value) in enumerate(self.dict_stat_r.items()):
        axs[idx].hist(value.flatten())

    plt.show()


#def plot_dist_activation_vgg16(self):
##        plt.subplot2grid((6,3),(0,0))
##        plt.hist(model.stat_a_conv1)
##        plt.subplot2grid((6,3),(0,1))
##        plt.hist(model.stat_a_conv1_1)
##        plt.subplot2grid((6,3),(1,0))
##        plt.hist(model.stat_a_conv2)
##        plt.subplot2grid((6,3),(1,1))
##        plt.hist(model.stat_a_conv2_1)
##        plt.subplot2grid((6,3),(2,0))
##        plt.hist(model.stat_a_conv3)
##        plt.subplot2grid((6,3),(2,1))
##        plt.hist(model.stat_a_conv3_1)
##        plt.subplot2grid((6,3),(2,2))
##        plt.hist(model.stat_a_conv3_1)
##        plt.subplot2grid((6,3),(3,0))
##        plt.hist(model.stat_a_conv4)
##        plt.subplot2grid((6,3),(3,1))
##        plt.hist(model.stat_a_conv4_1)
##        plt.subplot2grid((6,3),(3,2))
##        plt.hist(model.stat_a_conv4_2)
##        plt.subplot2grid((6,3),(4,0))
##        plt.hist(model.stat_a_conv5)
##        plt.subplot2grid((6,3),(4,1))
##        plt.hist(model.stat_a_conv5_1)
##        plt.subplot2grid((6,3),(4,2))
##        plt.hist(model.stat_a_conv5_2)
##        plt.subplot2grid((6,3),(5,0))
##        plt.hist(model.stat_a_fc1)
##        plt.subplot2grid((6,3),(5,1))
##        plt.hist(model.stat_a_fc2)
##        plt.subplot2grid((6,3),(5,2))
##        plt.hist(model.stat_a_fc3)
#
#    print(np.shape(self.dict_stat_w['fc1']))
#
#    plt.hist(self.dict_stat_w["fc1"][:,120],bins=100)
##    plt.show()


# SNN - postprocessing
#
#    # file write raw data
#    for n_name, n in self.list_neuron.items():
#        if not ('fc3' in n_name):
#            positive=tf.boolean_mask(n.first_spike_time,n.first_spike_time>0).numpy()
#
#            fname = './spike_time/spike-time'
#            if self.conf.f_load_time_const:
#                fname += '_train-'+str(self.conf.time_const_num_trained_data)+'_tc-'+str(self.conf.tc)+'_tw-'+str(self.conf.time_window)
#
#            fname += '_'+n_name+'.csv'
#            f = open(fname,'w')
#            wr = csv.writer(f)
#            wr.writerow(positive)
#            f.close()
#


#
def get_total_isi(self):
    isi_count=np.zeros(self.conf.time_step)

    for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
        if nn!='in' or nn!='fc3':
            isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
            isi_count_n.resize(self.conf.time_step)
            isi_count = isi_count + isi_count_n

    return isi_count

#
def f_out_isi(self,t):
    for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
        if nn!='in' or nn!='fc3':
            f_name = './isi/'+nn+'_'+self.conf.model_name+'_'+self.conf.input_spike_mode+'_'+self.conf.neural_coding+'_'+str(self.conf.time_step)+'.csv'

            if t==0:
                f = open(f_name,'w')
            else:
                f = open(f_name,'a')

            wr = csv.writer(f)

            array=n.isi.numpy().flatten()

            for i in range(len(array)):
                if array[i]!=0:
                    wr.writerow((i,n.isi.numpy().flatten()[i]))

            f.close()

#
def cal_entropy(self):
    #total_pattern=np.empty(len(self.list_layer_name))

    for il, length in enumerate(self.arr_length_entropy):
        #total_pattern=0
        #total_pattern=np.zeros(1)
        for idx_l, l in enumerate(self.list_layer_name):
            if l !='fc3':
                #print(self.dict_stat_w[l].shape)
                self.dict_stat_w[l][np.nonzero(self.dict_stat_w[l])]=1.0

                #print(self.dict_stat_w[l])

                #print(np.array2string(str(self.dict_stat_w[l]),max_line_width=4))
                #print(self.dict_stat_w[l].shape)

                num_words = self.dict_stat_w[l].shape[0]/length
                tmp = np.zeros((num_words,)+(self.dict_stat_w[l].shape[1:]))

                for idx in range(num_words):
                    for idx_length in range(length):
                        tmp[idx] += self.dict_stat_w[l][idx*length+idx_length]*np.power(2,idx_length)

                #print(tmp)

                #plt.hist(tmp.flatten())
                #plt.show()
                #print(tmp.shape)

                #print(np.histogram(tmp.flatten(),density=True))
                self.total_entropy[il,idx_l] += stats.entropy(np.histogram(tmp.flatten(),density=True)[0])
                #print(np.histogram(tmp.flatten()))

                #total_pattern = np.concatenate((total_pattern,tmp.flatten()))
                #print(total_pattern)

                #total_pattern=np.append(total_pattern,tmp.flatten())
                #total_pattern += (tmp.flatten())
                #total_pattern[idx_l]=tmp.flatten()

                self.total_entropy[il,-1] += self.total_entropy[il,idx_l]

        #self.total_entropy[il,-1]=np.histogram(total_pattern.flatten(),density=True)[0]
        #self.total_entropy[il,-1]=np.histogram(total_pattern,density=True)[0]

        #print(self.total_entropy[il,-1])



    #l='fc2'
    #print(self.dict_stat_w[l])
    #print(self.dict_stat_w[l].shape)

    #l='conv4'
    #print(self.dict_stat_w[l])
    #print(self.dict_stat_w[l].shape)



#
def get_total_spike_amp(self):
    assert False, 'not verified'
    spike_amp=np.zeros(self.spike_amp_kind)
    #print(range(0,self.spike_amp_kind)[::-1])
    #print(np.power(0.5,range(0,self.spike_amp_kind)))

    for idx_n, (nn, n) in enumerate(self.list_neuron.items()):
        if nn!='in' or nn!='fc3':
            spike_amp_n = np.histogram(n.out.numpy().flatten(),self.spike_amp_bin)
            #spike_amp = spike_amp + spike_amp_n[0]
            spike_amp += spike_amp_n[0]

            #isi_count_n = np.bincount(np.int32(n.isi.numpy().flatten()))
            #isi_count_n.resize(self.conf.time_step)
            #isi_count = isi_count + isi_count_n

    return spike_amp



