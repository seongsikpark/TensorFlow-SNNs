

import tensorflow as tf

from absl import flags
#from config_common import conf
conf = flags.FLAGS

import collections
import matplotlib.pyplot as plt

# global clock - time step
class GLB_CLK():
    def __init__(self):
        GLB_CLK.t = 1

    def __call__(self):
        GLB_CLK.t += 1

    def reset(self):
        GLB_CLK.t = 1

    def set(self, time):
        GLB_CLK.t = time

    def dec(self):
        GLB_CLK.t -= 1



# global configurations
class GLB():
    def __init__(self):
        GLB.model_compiled = False

    def model_compile_done_reset(self):
        GLB.model_compiled = False

    def model_compile_done(self):
        GLB.model_compiled = True


#
class GLB_PLOT():
    def __init__(self,title="",layer_names=[],idx_neurons=[],y=5,x=5):
        # TODO: parameterize
        self.figs, self.axes = plt.subplots(y, x, figsize=(12,10))

        #self.idx = 35291
        #self.idx = 7

        plt.suptitle(title)

        self.layers=layer_names
        self.idx_neurons = idx_neurons

        self.mark = 'bo'

        self.idx_current=0

        # subplot title
        for idx, layer_name in enumerate(self.layers):
            axe = self.axes.flatten()[idx]
            axe.set_title(layer_name+' neuron '+str(self.idx_neurons[idx]))


# TODO: move
#tf.compat.v1.app.flags.DEFINE_bool('_en_snn',False,'(internal) enable snn')
#tf.compat.v1.app.flags.DEFINE_bool('_bias_control',False,'(internal) bias control - SNN inference')
tf.compat.v1.app.flags.DEFINE_bool('_run_for_visual_debug',False,'(internal) run for visual debug')

def set_for_visual_debug(set):
    #flags.FLAGS._run_for_visual_debug = (set) and (flags.FLAGS.verbose_visual)
    conf._run_for_visual_debug = (set) and (conf.verbose_visual)

#def init_internal_config():
    #set_en_snn()
    #set_bias_control()

#def set_en_snn():
    #conf._en_snn = (conf.nn_mode == 'SNN' or conf.f_validation_snn)


#
glb = GLB()
glb_t = GLB_CLK()

#layers = ['conv1', 'conv1', 'conv1', 'conv1_1', 'conv1_1', 'conv1_1', 'conv2', 'conv2', 'conv2', 'conv3', 'conv3', 'conv3', 'conv4',
#          'conv4', 'conv4', 'conv5', 'conv5', 'conv5', 'fc1', 'fc1', 'fc1', 'fc2', 'fc2', 'fc2']
#idx_neurons = [0,20,5,0,20,5,10,13,30,13,14,9,1,7,9,2,5,9,0,2,10,3,4,10]


layers = ['conv1', 'conv1', 'conv1', 'conv1_1', 'conv1_1', 'conv1_1', 'conv2', 'conv2', 'conv2', 'conv3', 'conv3', 'conv3', 'conv4',
          'conv4', 'conv4', 'conv5', 'conv5', 'conv5', 'fc1', 'fc1', 'fc2', 'fc2', 'fc2', 'predictions', 'predictions']
idx_neurons = [0,20,5,0,20,5,10,13,30,13,14,9,1,7,9,2,5,9,0,2,3,4,10,4,5]


if conf.verbose_visual:
#if False:
    #
    if False:
        glb_plot = GLB_PLOT('plot',layers,idx_neurons)
        glb_plot_1 = GLB_PLOT('plot_1',layers,idx_neurons)
        glb_plot_2 = GLB_PLOT('plot_2',layers,idx_neurons)
        glb_plot_3 = GLB_PLOT('plot_3',layers,idx_neurons)
        glb_plot_1 = GLB_PLOT()

        # SNN training
        glb_plot_syn = GLB_PLOT(title='syn')
        glb_plot_bn = GLB_PLOT(title='bn')
        glb_plot_act = GLB_PLOT(title='act')
        glb_plot_kernel = GLB_PLOT(title='kernel')
        glb_plot_gradient_kernel = GLB_PLOT(title='gradient_kernel')
        glb_plot_gradient_gamma = GLB_PLOT(title='gradient_gamma')
        glb_plot_gradient_beta = GLB_PLOT(title='gradient_beta')
        glb_plot_gradient_bn = GLB_PLOT(title='gradient_bn')

        glb_plot_1x2 = GLB_PLOT([],[],1,2)


    glb_plot_kernel = GLB_PLOT(title='kernel')
    glb_plot_psp = GLB_PLOT(title='psp')

# integrated gradients
glb_ig_attributions = collections.OrderedDict()
glb_rand_vth = collections.OrderedDict()
glb_vth_search_err = collections.OrderedDict()
glb_vth_init = collections.OrderedDict()

glb_bias_comp = collections.OrderedDict()
glb_weight_comp = collections.OrderedDict()

class GLB_EPOCH():
    def __init__(self):
        GLB_EPOCH.epoch = 1
    def __call__(self):
        GLB_EPOCH.epoch += 1
    def reset(self):
        GLB_EPOCH.epoch = 1
    def set(self, epoch):
        GLB_EPOCH.epoch = epoch
    def dec(self):
        GLB_EPOCH.epoch -= 1
glb_epoch = GLB_EPOCH()