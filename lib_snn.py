import tensorflow as tf
import tensorflow.contrib.eager as tfe
import sys

import numpy as np

class Neuron(tf.layers.Layer):
    def __init__(self,dim,n_type,fan_in,conf,neural_coding,**kwargs):
        super(Neuron, self).__init__(name="")


        self.dim = dim
        self.n_type = n_type
        self.fan_in = fan_in

        self.conf = conf

        self.neural_coding=neural_coding

    def build(self, _):

        if self.n_type == 'IN':
            init_vth = self.conf.n_in_init_vth
        else:
            init_vth = self.conf.n_init_vth

        self.vth_init = self.add_variable("vth_init",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(init_vth),trainable=False)
        #self.vth_init = tfe.Variable(init_vth)
        self.vth = self.add_variable("vth",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(init_vth),trainable=False)

        self.vmem = self.add_variable("vmem",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.conf.n_init_vinit),trainable=False)

        self.out = self.add_variable("out",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        if self.conf.f_isi:
            self.last_spike_time = self.add_variable("last_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
            self.isi = self.add_variable("isi",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.spike_counter_int = self.add_variable("spike_counter_int",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
        self.spike_counter = self.add_variable("spike_counter",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        if self.conf.f_refractory:
            self.refractory = self.add_variable("refractory",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)


    def call(self,inputs,t):
        
        #print('neuron call')

        self.reset_each_time()

        # reshape
        #vth = tf.reshape(self.vth,self.dim)
        #vmem = tf.reshape(self.vmem,self.dim)
        #out = tf.reshape(self.out,self.dim)
        #inputs = tf.reshape(inputs,self.dim)

        #if inputs.shape[0] != 1:
        #    print('not supported batch mode in SNN test mode')
        #    sys.exit(1)
        #else:
        #    inputs = tf.reshape(inputs,self.vmem.shape)
        #inputs = tf.reshape(inputs,[-1]+self.vmem.shape[1:])

        # run_fwd
        run_type = {
            'IN': self.run_type_in,
            'IF': self.run_type_if,
            'LIF': self.run_type_lif,
            'OUT': self.run_type_out
        }[self.n_type](inputs,t)

        #print(self.vmem)
        #print(self.out)
        #print(self.spike_counter)

        out_ret = tf.reshape(self.out,self.dim)

        return out_ret

    # reset - time step
    def reset_each_time(self):
        self.reset_out()

    # reset - sample
    def reset(self):
        #print('reset neuron')
        self.reset_vmem()
        self.reset_out()
        self.reset_spike_count()
        self.reset_vth()
        if self.conf.f_isi:
            self.last_spike_time = tf.zeros(self.last_spike_time.shape)
            self.isi = tf.zeros(self.isi.shape)
        if self.conf.f_refractory:
            self.refractory = tf.zeros(self.refractory.shape)

    def reset_spike_count(self):
        self.spike_counter = tf.zeros(self.out.shape)
        self.spike_counter_int = tf.zeros(self.out.shape)

    #
    def reset_vmem(self):
        self.vmem = tf.constant(self.conf.n_init_vinit,tf.float32,self.vmem.shape)

    #
    def reset_out(self):
        self.out = tf.zeros(self.out.shape)

    def reset_vth(self):
        self.vth = self.vth_init

    def input_spike_real(self,inputs,t):
        if self.conf.neural_coding=="WEIGHTED_SPIKE":
            self.out=inputs/self.conf.p_ws
        else:
            self.out=inputs

    def input_spike_poission(self,inputs,t):
        # Poission input
        vrand = tf.random_uniform(self.vmem.shape,minval=0.0,maxval=1.0,dtype=tf.float32)

        f_fire = inputs>=vrand

        self.out = tf.where(f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))

    def input_spike_weighted_spike(self,inputs,t):
        # weighted synpase input
        t_mod = (int)(t%8)
        if t_mod == 0:
            self.vmem = inputs
            self.vth = tf.constant(0.5,tf.float32,self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth,0.5)

        f_fire = self.vmem >= self.vth

        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem,self.out)

    def input_spike_proposed(self,inputs,t):
        # proposed method
        #t_mod = (int)(t%8)
        if t == 0:
            self.vmem = inputs
            #self.vth=self.vth_init

        #print(self.vmem)
        #print(self.vth)

        f_fire = self.vmem >= self.vth

        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))
        #self.out = tf.zeros(self.out.shape)

        self.vmem = tf.subtract(self.vmem,self.out)

        self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)

        # repeat input
        #print(tf.reduce_max(self.out))
        if tf.equal(tf.reduce_max(self.out),0.0):
            self.vmem = inputs



    # (min,max) = (0.0,1.0)
    def input_spike_gen(self,inputs,t):
        input_spike_mode ={
            'REAL': self.input_spike_real,
            'POISSON': self.input_spike_poission,
            'WEIGHTED_SPIKE': self.input_spike_weighted_spike,
            'PROPOSED': self.input_spike_proposed
        }

        input_spike_mode[self.conf.input_spike_mode](inputs,t)


    #
    def leak(self):
        self.vmem = tf.multiply(self.vmem,0.7)

    #
    def cal_isi(self, f_fire, t):
        f_1st_spike = self.last_spike_time == 0.0
        f_isi_update = np.logical_and(f_fire, np.logical_not(f_1st_spike))

        self.isi = tf.where(f_isi_update,tf.constant(t,tf.float32,self.isi.shape)-self.last_spike_time,np.zeros(self.isi.shape))

        self.last_spike_time = tf.where(f_fire,tf.constant(t,tf.float32,self.last_spike_time.shape),self.last_spike_time)


    #
    def integration(self,inputs,t):
        # weighted synapse
        #t_mod = (int)(t%8)
        #inputs = inputs*1.0/np.power(2,t_mod+1)

        #
        self.vmem = tf.add(self.vmem,inputs)
        if self.conf.f_positive_vmem:
            self.vmem = tf.maximum(self.vmem,tf.constant(0.0,tf.float32,self.vmem.shape))

    def cal_refractory(self,f_fire):
        f_refractory_update = np.logical_and(np.not_equal(self.vth-self.vth_init,0.0),np.logical_not(f_fire))
        refractory_update = 2.0*np.log2(self.vth/self.vth_init)

        self.refractory = tf.maximum(self.refractory-1,tf.constant(0.0,tf.float32,self.refractory.shape))

        self.refractory = tf.where(f_refractory_update,refractory_update,self.refractory)

        #print(tf.reduce_max(self.vth))
        #print(self.vth_init)
        #print(np.not_equal(self.vth,self.vth_init))
        #print(np.logical_not(f_fire))
        #print(f_refractory_update)
        #self.refractory = tf.where(f_fire,tf.constant(0.0,tf.float32,self.refractory.shape),self.refractory)
        #print(self.refractory)

        #print(tf.reduce_max(np.log2(self.vth/self.vth_init)))




    #
    def fire(self,t):
        #self.fire(idx for (idx,f_fire) in enumerate(self.vmem>=self.vth) if f_fire==True)

        f_fire = self.vmem >= self.vth

        #print('f_fire')
        #print(f_fire)

        # vmem -> vrest
        # reset to zero
        #self.vmem = tf.where(f_fire,tf.constant(self.conf.n_init_vreset,tf.float32,self.vmem.shape,self.vmem)
        # reset by subtraction
        self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)

        self.out = tf.where(f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))

        #print(self.out)

        if self.conf.f_isi:
            self.cal_isi(f_fire,t)


    def fire_weighted_spike(self,t):
        # weighted synpase input
        t_mod = (int)(t%self.conf.p_ws)
        if t_mod == 0:
            self.vth = tf.constant(0.5,tf.float32,self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth,0.5)

        if self.conf.f_refractory:
            f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
        else:
            f_fire = self.vmem >= self.vth

        if self.conf.f_refractory:
            print('fire_weighted_spike, refractory: not implemented yet')

        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem,self.out)

        if self.conf.f_isi:
            self.cal_isi(f_fire,t)


    #
    def fire_proposed(self,t):
        if self.conf.f_refractory:
            f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
        else:
            f_fire = self.vmem >= self.vth


        #print(f_fire)
        #print(np.equal(self.refractory,0.0))
        #print(self.refractory)


        # reset by subtraction
        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem,self.out)

        if self.conf.f_refractory:
            self.cal_refractory(f_fire)


        # exp increasing order
        self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*1.5,self.vth_init)
        # exp decreasing order
        #self.vth = tf.where(f_fire,self.vth*0.5,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*0.9,self.vth_init)

        if self.conf.f_isi:
            self.cal_isi(f_fire,t)


    def fire_type_out(self, t):
        f_fire = self.vmem >= self.vth

        self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)

        self.out = tf.where(f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))


        #self.isi = tf.where(f_fire,tf.constant(t,tf.float32,self.isi.shape)-self.last_spike_time,self.isi)
        #self.last_spike_time = tf.where(f_fire,tf.constant(t,tf.float32,self.last_spike_time.shape),self.last_spike_time)


    def count_spike(self, t):
        #print(np.nonzero(self.out.numpy()))
        #print(len(np.nonzero(self.out.numpy())))
        #print(np.count_nonzero(self.out.numpy()))

        #self.spike_counter_int = self.spike_counter_int + np.count_nonzero(self.out.numpy(),axis=0)

        f_fire = np.not_equal(self.out,0.0)
        self.spike_counter_int = tf.where(f_fire,self.spike_counter_int+1.0,self.spike_counter_int)
        self.spike_counter = self.spike_counter + self.out.numpy()

    # run fwd
    def run_type_in(self,inputs,t):
        #print('run_type_in')
        self.input_spike_gen(inputs,t)
        self.count_spike(t)

    def run_type_if(self,inputs,t):
        #print('run_type_if')
        self.integration(inputs,t)

        neural_coding={
            'RATE': self.fire,
            'WEIGHTED_SPIKE': self.fire_weighted_spike,
            'PROPOSED': self.fire_proposed
        }

        neural_coding[self.neural_coding](t)

        #self.fire()
        #self.fire_proposed()

        self.count_spike(t)

    def run_type_lif(self,inputs,t):
        #print('run_type_lif')
        self.leak()
        self.integration(inputs,t)
        self.fire(t)
        self.count_spike(t)

    def run_type_out(self,inputs,t):
        self.integration(inputs,t)
        #self.fire_type_out(t)

    def set_vth(self,vth):
        #self.vth = self.vth.assign(vth)
        self.vth.assign(vth)

    def get_spike_count(self):
        spike_count = tf.reshape(self.spike_counter,self.dim)
        return spike_count

    def get_spike_count_int(self):
        spike_count_int = tf.reshape(self.spike_counter_int,self.dim)
        return spike_count_int

    def get_isi(self):
        return self.isi



