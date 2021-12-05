
import matplotlib.pyplot as plt

# global clock - time step
class GLB_CLK():
    def __init__(self):
        GLB_CLK.t = 1

    def __call__(self):
        GLB_CLK.t += 1

    def reset(self):
        GLB_CLK.t = 1


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
    def __init__(self,layer_names=[],idx_neurons=[]):
        # TODO: parameterize
        self.figs, self.axes = plt.subplots(5, 5, figsize=(12,10))

        #self.idx = 35291
        #self.idx = 7

        self.layers=layer_names
        self.idx_neurons = idx_neurons

        self.mark = 'bo'

        # subplot title
        for idx, layer_name in enumerate(self.layers):
            axe = self.axes.flatten()[idx]
            axe.set_title(layer_name+' neuron '+str(self.idx_neurons[idx]))




#
glb = GLB()
glb_t = GLB_CLK()

layers = ['conv1', 'conv1', 'conv1', 'conv1_1', 'conv1_1', 'conv1_1', 'conv2', 'conv2', 'conv2', 'conv3', 'conv3', 'conv3', 'conv4',
          'conv4', 'conv4', 'conv5', 'conv5', 'conv5', 'fc1', 'fc1', 'fc1', 'fc2', 'fc2', 'fc2']
idx_neurons = [0,20,5,0,20,5,10,13,30,13,14,9,1,7,9,2,5,9,0,2,10,3,4,10]

glb_plot = GLB_PLOT(layers,idx_neurons)
glb_plot_1 = GLB_PLOT(layers,idx_neurons)
glb_plot_2 = GLB_PLOT(layers,idx_neurons)
#glb_plot_1 = GLB_PLOT()
