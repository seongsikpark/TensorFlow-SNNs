
'''
 This code is written based on www.mathformachines.com
'''

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow import keras
import keras
#from tensorflow.keras import callbacks, layers
from keras import callbacks, layers


from sklearn.decomposition import PCA

#
class RandomCoordinates(object):
    def __init__(self,origin):
        self._origin = origin
        self._v0 = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )
        self._v1 = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )

    def __call__(self, a, b):
        return [a*w0 + b*w1 + wc for w0, w1, wc in zip(self._v0,self._v1,self._origin)]

def normalize_weights(weights, origin):
    return [ w*np.linalg.norm(wc)/np.linalg.norm(w) for w, wc in zip(weights, origin)]




#
def vectorize_weights(weights):
    vec = [w.flatten() for w in weights]
    vec = np.hstack(vec)
    return vec


#
def vectorize_weight_list(weight_list):
    vec_list = []
    for weights in weight_list:
        vec_list.append(vectorize_weights(weights))

    weight_matrix = np.column_stack(vec_list)
    return weight_matrix

def shape_weight_matrix_like(weight_matrix, example):
    weight_vecs = np.hsplit(weight_matrix, weight_matrix.shape[1])
    sizes = [v.size for v in example]
    shapes = [v.shape for v in example]
    weight_list = []
    for net_weights in weight_vecs:
        vs = np.split(net_weights, np.cumsum(sizes))[:-1]
        vs = [v.reshape(s) for v, s in zip(vs, shapes)]
        weight_list.append(vs)
    return weight_list


#
def get_path_components(training_path, n_components=2):
    # Vectorize network weights
    weights_matrix = vectorize_weight_list(training_path)

    # Create components
    pca = PCA(n_components=2, whiten=True)
    components = pca.fit_transform(weights_matrix)

    # Reshape to fit network
    example = training_path[0]
    weight_list = shape_weight_matrix_like(components, example)
    return pca, weight_list

#
def weights_to_coordinates(coords, training_path):
    # project training path onto the first two principal components using pseudoinverse

    components = [coords.v0, coords.v1]
    comp_matrix = vectorize_weight_list(components)

    # pseudoinverse
    comp_matrix_i = np.linalg.pinv(comp_matrix)

    # origin vector
    w_c = vectorize_weights(training_path[-1])

    # center the weights on the training path and project onto components
    coord_path = np.array([comp_matrix_i @ (vectorize_weights(weights)-w_c)for weights in training_path])

    return coord_path

#
def plot_training_path(coords, training_path, ax=None, end=None, **kwargs):
    path = weights_to_coordinates(coords, training_path)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    colors = range(path.shape[0])
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0,end)
    ax.scatter(path[:,0],path[:,1],s=4,c=colors,cmap="cividis",norm=norm)
    return ax

#
class LossSurface(object):
    def __init__(self, model, inputs, outputs):
    #def __init__(self, model, ds):
        self._model = model
        self._inputs = inputs
        self._outputs = outputs
        #self.ds = ds

    def compile(self, range, points, coords):
        a_grid = tf.linspace(-1.0, 1.0, num=points) ** 3 * range
        b_grid = tf.linspace(-1.0, 1.0, num=points) ** 3 * range
        loss_grid = np.empty([len(a_grid),len(b_grid)])

        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                self._model.set_weights(coords(a,b))
                loss = self._model.test_on_batch(self._inputs,self._outputs,return_dict=True)["loss"]
                #loss = self._model.test_on_batch(self.ds,return_dict=True)["loss"]
                loss_grid[j,i] = loss

        self._model.set_weights(coords.origin)
        self._a_grid = a_grid
        self._b_grid = b_grid
        self._loss_grid = loss_grid

    def plot(self, range=1.0, points=24, levels=20, ax=None, **kwargs):
        xs = self._a_grid
        ys = self._b_grid
        zs = self._loss_grid

        if ax is None:
            _, ax = plt.subplots(**kwargs)
            ax.set_title("The Loss Surface")
            ax.set_aspect("equal")

            # set levels
            min_loss = zs.min()
            max_loss = zs.max()
            levels = tf.exp(tf.linspace(tf.math.log(min_loss,),tf.math.log(max_loss),num=levels))

            # Contour plot
            cs = ax.contour(xs,ys,zs,levels=levels,cmap='magma',linewidths=0.75,
                            norm=matplotlib.colors.LogNorm(vmin=min_loss,vmax=max_loss*2.0))
            ax.clabel(cs, inline=True, fontsize=8, fmt="%1.2f")

            return ax

#
class PCACoordinates(object):
    def __init__(self, training_path):
        origin = training_path[-1]
        self.pca, self.components = get_path_components(training_path)
        self.set_origin(origin)

    def __call__(self, a, b):
        return [a*w0 + b*w1 + wc for w0, w1, wc in zip(self.v0,self.v1,self.origin)]

    def set_origin(self, origin, renorm=True):
        self.origin = origin
        if renorm:
            self.v0 = normalize_weights(self.components[0],origin)
            self.v1 = normalize_weights(self.components[1],origin)


