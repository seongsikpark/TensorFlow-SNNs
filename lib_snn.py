import tensorflow as tf
import tensorflow.contrib.eager as tfe

#import tensorflow_probability as tfp

import sys

import numpy as np

import matplotlib.pyplot as plt

class Neuron(tf.layers.Layer):
    def __init__(self,dim,n_type,fan_in,conf,neural_coding,depth=0,n_name='',**kwargs):
        #super(Neuron, self).__init__(name="")
        super(Neuron, self).__init__()

        self.dim = dim
        self.n_type = n_type
        self.fan_in = fan_in

        self.conf = conf

        self.neural_coding=neural_coding

        self.n_name = n_name

        #self.zeros = np.zeros(self.dim,dtype=np.float32)
        #self.zeros = tf.constant(0.0,shape=self.dim,dtype=tf.float32)
        #self.fires = np.full(self.dim, self.conf.n_in_init_vth,dtype=np.float32)

        self.zeros = tf.zeros(self.dim,dtype=tf.float32)
        self.fires = tf.constant(self.conf.n_in_init_vth,shape=self.dim,dtype=tf.float32)

        self.depth = depth

        if self.conf.f_record_first_spike_time:
            self.init_first_spike_time = -1.0



        if self.conf.neural_coding=='TEMPORAL':
            self.time_const_init_fire = self.conf.tc
            self.time_const_init_integ = self.conf.tc

            self.time_delay_init_fire = 0.0
            self.time_delay_init_integ = 0.0


        #self.spike_counter = tf.Variable(name="spike_counter",dtype=tf.float32,initial_value=tf.zeros(self.dim,dtype=tf.float32),trainable=False)
        #self.spike_counter_int = tf.Variable(name="spike_counter_int",dtype=tf.float32,initial_value=tf.zeros(self.dim,dtype=tf.float32),trainable=False)
        #self.f_fire = tf.Variable(name='f_fire', dtype=tf.bool, initial_value=tf.constant(False,dtype=tf.bool,shape=self.dim),trainable=False)

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
        #self.out = self.add_variable("out",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=True)

        if self.conf.f_isi:
            self.last_spike_time = self.add_variable("last_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
            self.isi = self.add_variable("isi",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.spike_counter_int = self.add_variable("spike_counter_int",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)
        self.spike_counter = self.add_variable("spike_counter",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        self.f_fire = self.add_variable("f_fire",shape=self.dim,dtype=tf.bool,trainable=False)

        if self.conf.f_tot_psp:
            self.tot_psp = self.add_variable("tot_psp",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        if self.conf.f_refractory:
            self.refractory = self.add_variable("refractory",shape=self.dim,dtype=tf.float32,initializer=tf.zeros_initializer,trainable=False)

        #self.depth = self.add_variable("depth",shape=self.dim,dtype=tf.int32,initializer=tf.zeros_initializer,trainable=False)

        #if self.conf.neural_coding=='TEMPORAL':
        if self.conf.f_record_first_spike_time:
            self.first_spike_time=self.add_variable("first_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.init_first_spike_time),trainable=False)

        #if self.conf.f_train_time_const:
        if self.conf.neural_coding=='TEMPORAL':
            #self.time_const=self.add_variable("time_const",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.conf.tc),trainable=False)
            self.time_const_integ=self.add_variable("time_const_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_integ),trainable=False)
            self.time_const_fire=self.add_variable("time_const_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_fire),trainable=False)
            self.time_delay_integ=self.add_variable("time_delay_integ",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_integ),trainable=False)
            self.time_delay_fire=self.add_variable("time_delay_fire",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_fire),trainable=False)


    def call(self,inputs,t):
        
        #print('neuron call')

        #self.reset_each_time()

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

        #out_ret = tf.reshape(self.out,self.dim)
        out_ret = self.out

        return out_ret

    # reset - time step
    def reset_each_time(self):
        self.reset_out()

    # reset - sample
    def reset(self):
        #print('reset neuron')
        self.reset_vmem()
        #self.reset_out()
        self.reset_spike_count()
        self.reset_vth()

        if self.conf.f_tot_psp:
            self.reset_tot_psp()
        if self.conf.f_isi:
            self.last_spike_time = tf.zeros(self.last_spike_time.shape)
            self.isi = tf.zeros(self.isi.shape)
        if self.conf.f_refractory:
            self.refractory = tf.zeros(self.refractory.shape)

        if self.conf.f_record_first_spike_time:
            self.reset_first_spike_time()

    def reset_spike_count(self):
        #self.spike_counter = tf.zeros(self.dim)
        #self.spike_counter_int = tf.zeros(self.dim)

        self.spike_counter = tf.zeros(self.dim,dtype=tf.float32)
        self.spike_counter_int = tf.zeros(self.dim,dtype=tf.float32)

        #self.spike_counter.assign(self.zeros)
        #self.spike_counter.assign(tf.zeros(self.dim))
        #self.spike_counter_int = tf.zeros(self.out.shape)
        #self.spike_counter_int.assign(tf.zeros(self.dim))

    #
    def reset_vmem(self):
        self.vmem = tf.constant(self.conf.n_init_vinit,tf.float32,self.vmem.shape)

    #
    def reset_tot_psp(self):
        self.tot_psp = tf.zeros(tf.shape(self.tot_psp))

    #
    def reset_out(self):
        self.out = tf.zeros(self.out.shape)

    def reset_vth(self):
        self.vth = self.vth_init

    #
    def reset_first_spike_time(self):
        self.first_spike_time=tf.constant(self.init_first_spike_time,shape=self.first_spike_time.shape,dtype=tf.float32)

    ##
    def set_vth_temporal_kernel(self,t):
        # exponential decay
        #self.vth = tf.constant(tf.exp(-float(t)/self.conf.tc),tf.float32,self.out.shape)
        #self.vth = tf.constant(tf.exp(-t/self.conf.tc),tf.float32,self.out.shape)
        time = tf.subtract(t,self.time_delay_fire)
        self.vth = tf.constant(tf.exp(-time/self.time_const_fire),tf.float32,self.out.shape)

        # polynomial
        #self.vth = tf.constant(tf.add(-tf.pow(t/self.conf.tc,2),1.0),tf.float32,self.out.shape)


    ##
    def input_spike_real(self,inputs,t):
        if self.conf.neural_coding=="WEIGHTED_SPIKE":
            self.out=tf.truediv(inputs,self.conf.p_ws)
        else:
            self.out=inputs

    def input_spike_poission(self,inputs,t):
        # Poission input
        vrand = tf.random_uniform(self.vmem.shape,minval=0.0,maxval=1.0,dtype=tf.float32)

        self.f_fire = inputs>=vrand

        self.out = tf.where(self.f_fire,self.fires,self.zeros)
        #self.out = tf.where(self.f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))

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

    def input_spike_burst(self,inputs,t):
        if t == 0:
            self.vmem = inputs

        f_fire = self.vmem >= self.vth

        self.out = tf.where(f_fire,self.vth,tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem,self.out)

        self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)

        # repeat input
        if tf.equal(tf.reduce_max(self.out),0.0):
            self.vmem = inputs

    def input_spike_temporal(self,inputs,t):

        if t == 0:
            self.vmem = inputs


        #kernel = self.temporal_kernel(t)
        #self.vth = tf.constant(kernel,tf.float32,self.out.shape)
        #self.vth = tf.constant(tf.exp(-t/self.conf.tc),tf.float32,self.out.shape)
        #self.vth = self.temporal_kernel(t)
        self.set_vth_temporal_kernel(t)
        #print(self.vth[0,1,1,1])

        #print("input_spike_temporal: depth: "+str(self.depth)+", t_glb: "+str(t)+", t_loc: "+str(t)+", kernel: "+str(self.vth[0,0,0,0].numpy()))


        if self.conf.f_refractory:
            self.f_fire = (self.vmem >= self.vth) & \
                          tf.equal(self.refractory,tf.constant(0.0,tf.float32,self.refractory.shape))
                        #(self.vth >= tf.constant(10**(-5),tf.float32,self.vth.shape) ) & \

        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10**(-5))

        #print(f_fire)
        #print(self.vth >= 10**(-1))

        # reset by subtraction
        #self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        #self.vmem = tf.subtract(self.vmem,self.out)

        # reset by zero
        self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)





        if self.conf.f_refractory:
            self.cal_refractory_temporal(self.f_fire)


        #self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)

        # repeat input
        #print(tf.reduce_max(self.out))
        #if tf.equal(tf.reduce_max(self.out),0.0):
        #    self.vmem = inputs

    def spike_dummy_input(self,inputs,t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False,tf.bool,self.f_fire.shape)

    def spike_dummy_fire(self,t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False,tf.bool,self.f_fire.shape)

    # (min,max) = (0.0,1.0)
    def input_spike_gen(self,inputs,t):
        input_spike_mode ={
            'REAL': self.input_spike_real,
            'POISSON': self.input_spike_poission,
            'WEIGHTED_SPIKE': self.input_spike_weighted_spike,
            'BURST': self.input_spike_burst,
            'TEMPORAL': self.input_spike_temporal if t < self.conf.time_fire_duration else self.spike_dummy_input
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

        #f_run_int_temporal = (t >= (self.depth-1)*self.conf.time_window and t < (self.depth)*self.conf.time_window) or (t==0)   # t==0 : for bias integration
        #f_run_int_temporal = (t >= (self.depth-1)*self.conf.time_fire_start and t < (self.depth)*self.conf.time_window) or (t==0)   # t==0 : for bias integration

        t_int_s = (self.depth-1)*self.conf.time_fire_start
        t_int_e = t_int_s + self.conf.time_fire_duration


        f_run_int_temporal = (t >= t_int_s and t < t_int_e) or (t==0)   # t==0 : for bias integration

        # intergation
        #{
        #    'TEMPORAL': self.integration_temporal if f_run_int_temporal else lambda inputs, t : None
        #}.get(self.neural_coding, self.integration) (inputs,t)

        #self.integration(inputs,t)

        #if(self.depth==16):
        #    print(self.vmem.numpy())


        if f_run_int_temporal:
            self.integration_temporal(inputs,t)


        #
        #if f_run_int_temporal:
        #    if self.depth < 3 and self.depth > 0:
        #        self.integration_temporal(inputs,t)
        #    else:
        #        self.integration_default(inputs,t)

        #self.integration_default(inputs,t)

        ########################################
        # common
        ########################################

        #
        if self.conf.f_positive_vmem:
            self.vmem = tf.maximum(self.vmem,tf.constant(0.0,tf.float32,self.vmem.shape))

        #
        if self.conf.f_tot_psp:
            self.tot_psp = tf.add(self.tot_psp, inputs)


    #
    def integration_default(self,inputs,t):
#        if self.depth >= 3:
#            if t!=0:
#                #psp = tf.multiply(inputs,tf.constant(0.5,tf.float32,self.vmem.shape))
#                #psp = tf.multiply(inputs,tf.constant(2.0,dtype=tf.float32,shape=self.vmem.shape))
#                #psp = tf.multiply(inputs,tf.constant(1.0,dtype=tf.float32,shape=self.vmem.shape))
#
#                time = float(t-(self.depth-1)*self.conf.time_window)
#                receptive_kernel = tf.constant(tf.exp(-time/self.conf.tc),tf.float32,self.vmem.shape)
#
#                psp = tf.multiply(inputs,receptive_kernel)
#
#            else:
#                psp = inputs
#
#
#            self.vmem = tf.add(self.vmem,psp)
#        else:
#            self.vmem = tf.add(self.vmem,inputs)

        self.vmem = tf.add(self.vmem,inputs)

    #
    def integration_temporal(self,inputs,t):
        if(t==0) :
            time = 0
        else :
            #time = float(t-(self.depth-1)*self.conf.time_window)
            time = t-(self.depth-1)*self.conf.time_fire_start
            time = time - self.time_delay_integ
        #time = tf.zeros(self.vmem.shape)

        #receptive_kernel = tf.constant(tf.exp(-time/self.conf.tc),tf.float32,self.vmem.shape)

        receptive_kernel = tf.constant(tf.exp(-time/self.time_const_integ),tf.float32,self.vmem.shape)

        #print("integration_temporal: depth: "+str(self.depth)+", t_glb: "+str(t)+", t_loc: "+str(time)+", kernel: "+str(receptive_kernel[0,0,0,0].numpy()))

        #
        psp = tf.multiply(inputs,receptive_kernel)
        #print(inputs[0,0,0,1])
        #print(psp[0,0,0,1])


        #addr=0,10,10,0
        #print("int: glb {}: loc {} - in {:0.3f}, kernel {:.03f}, psp {:0.3f}".format(t, time,inputs[addr],receptive_kernel[addr],psp[addr]))

        #
        #self.vmem = tf.add(self.vmem,inputs)
        self.vmem = tf.add(self.vmem,psp)




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

    def cal_refractory_temporal(self,f_fire):
        #self.refractory = tf.where(f_fire,tf.constant(self.conf.time_step,tf.float32,self.refractory.shape),self.refractory)
        self.refractory = tf.where(f_fire,tf.constant(10000.0,tf.float32,self.refractory.shape),self.refractory)

    #
    def fire(self,t):
        #self.fire(idx for (idx,f_fire) in enumerate(self.vmem>=self.vth) if f_fire==True)

        #self.f_fire.assign(self.vmem >= self.vth)
        self.f_fire = self.vmem >= self.vth

        #f_fire_new = np.where(self.vmem >= self.vth)
        #print(tf.shape(f_fire))
        #print(tf.shape(f_fire_new))
        #print(f_fire_new)

        #print('f_fire')
        #print(f_fire)

        #self.out = tf.zeros(self.vmem.shape,dtype=tf.float32)
        #self.out = self.zeros
        #self.out = tf.where(f_fire,tf.constant(self.conf.n_in_init_vth,tf.float32,self.out.shape),self.out)

        #self.out = tf.where(f_fire,tf.constant(self.conf.n_in_init_vth,tf.float32,self.out.shape),self.zeros)

        #self.out = tf.where(f_fire,self.fires,self.zeros)


        #self.out = tf.where(self.f_fire,tf.constant(self.conf.n_in_init_vth,tf.float32,self.dim),tf.zeros(self.dim))
        self.out = tf.where(self.f_fire,self.fires,self.zeros)

        # vmem -> vrest
        # reset to zero
        #self.vmem = tf.where(f_fire,tf.constant(self.conf.n_init_vreset,tf.float32,self.vmem.shape,self.vmem)
        # reset by subtraction
        #self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)
        self.vmem = tf.subtract(self.vmem,self.out)


        #print(self.out)

        if self.conf.f_isi:
            self.cal_isi(self.f_fire,t)


    def fire_weighted_spike(self,t):
        # weighted synpase input
        t_mod = (int)(t%self.conf.p_ws)
        if t_mod == 0:
            self.vth = tf.constant(0.5,tf.float32,self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth,0.5)

        if self.conf.f_refractory:
            #self.f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
            self.f_fire = tf.logical_and(self.vmem >= self.vth, np.equal(self.refractory,0.0))
        else:
            self.f_fire = self.vmem >= self.vth

        if self.conf.f_refractory:
            print('fire_weighted_spike, refractory: not implemented yet')

        self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem,self.out)

        if self.conf.f_isi:
            self.cal_isi(self.f_fire,t)


    #
    def fire_burst(self,t):
        if self.conf.f_refractory:
            #self.f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
            self.f_fire = tf.logical_and(self.vmem >= self.vth, np.equal(self.refractory,0.0))
        else:
            self.f_fire = self.vmem >= self.vth


        #print(f_fire)
        #print(np.equal(self.refractory,0.0))
        #print(self.refractory)


        # reset by subtraction
        self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem,self.out)

        if self.conf.f_refractory:
            self.cal_refractory(f_fire)


        # exp increasing order
        self.vth = tf.where(self.f_fire,self.vth*2.0,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*1.5,self.vth_init)
        # exp decreasing order
        #self.vth = tf.where(f_fire,self.vth*0.5,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*0.9,self.vth_init)

        if self.conf.f_isi:
            self.cal_isi(self.f_fire,t)

    #
    def fire_temporal(self,t):
        #time = t-self.depth*self.conf.time_window
        time = t-self.depth*self.conf.time_fire_start
        #print("fire_temporal: depth: "+str(self.depth)+", t_glb: "+str(t)+", t_loc: "+str(time))

        self.set_vth_temporal_kernel(time)

        if self.conf.f_refractory:
            #self.f_fire = (self.vmem >= self.vth) & \
            #                tf.equal(self.refractory,tf.zeros(self.refractory.shape))
            self.f_fire = (self.vmem >= self.vth) & \
                              tf.equal(self.refractory,tf.constant(0.0,tf.float32,self.refractory.shape))



        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10**(-5))

#        if time >= 0 and time < self.conf.time_window:
#        #if time >= 0 and time < self.conf.tc*4:
#        #if time > 0:
#            #print("t: "+str(t)+", time: "+str(time))
#
#            #kernel = self.temporal_kernel(t)
#            #self.vth = tf.constant(kernel,tf.float32,self.vmem.shape)
#            #self.vth = tf.constant(tf.exp(-time/self.conf.tc),tf.float32,self.vmem.shape)
#            #self.vth = self.temporal_kernel(t)
#            self.set_vth_temporal_kernel(time)
#
#            #self.f_fire = self.vmem >= self.vth
#
#            if self.conf.f_refractory:
#                self.f_fire = (self.vmem >= self.vth) & \
#                          tf.equal(self.refractory,tf.zeros(self.refractory.shape))
#            else:
#                self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10**(-5))
#        else :
#            self.f_fire = tf.constant(False,tf.bool,self.f_fire.shape)

        # reset by zero
        self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)

        # reset by subtraction
        #self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        #self.vmem = tf.subtract(self.vmem,self.out)


#        #
#        if self.depth < 2 and self.depth > 0:
#
#            #out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
#            #self.out = tf.multiply(out,tf.constant(2.0,tf.float32,self.out.shape))
#            #self.out = tf.multiply(out,tf.constant(1.0,dtype=tf.float32,shape=self.out.shape))
#            #self.vmem = tf.subtract(self.vmem,out)
#            #self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
#            #self.vmem = tf.where(self.f_fire,tf.zeros(self.out.shape),self.vmem)
#        elif self.depth >= 2 :
#            self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
#
#            #out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
#            #self.out = tf.multiply(out,tf.constant(2.0,tf.float32,self.out.shape))
#
#        else:


        #addr=0,10,10,0
        #print("fire: glb {}: loc {} - vth {:0.3f}, kernel {:.03f}, out {:0.3f}".format(t, time,self.vmem[addr],self.vth[addr],self.out[addr]))



        if self.conf.f_refractory:
            self.cal_refractory_temporal(self.f_fire)

        # exp increasing order
        #self.vth = tf.where(self.f_fire,self.vth*2.0,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*1.5,self.vth_init)
        # exp decreasing order
        #self.vth = tf.where(f_fire,self.vth*0.5,self.vth_init)
        #self.vth = tf.where(f_fire,self.vth*0.9,self.vth_init)




    def fire_type_out(self, t):
        f_fire = self.vmem >= self.vth

        self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)

        self.out = tf.where(f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))


        #self.isi = tf.where(f_fire,tf.constant(t,tf.float32,self.isi.shape)-self.last_spike_time,self.isi)
        #self.last_spike_time = tf.where(f_fire,tf.constant(t,tf.float32,self.last_spike_time.shape),self.last_spike_time)


    #
    def count_spike(self, t):
        {
            'TEMPORAL': self.count_spike_temporal
        }.get(self.neural_coding, self.count_spike_default) (t)


    def count_spike_default(self, t):
        self.spike_counter_int = tf.where(self.f_fire,self.spike_counter_int+1.0,self.spike_counter_int)
        self.spike_counter = tf.add(self.spike_counter, self.out)

    def count_spike_temporal(self, t):
        self.spike_counter_int = tf.add(self.spike_counter_int, self.out)
        self.spike_counter = tf.where(self.f_fire, tf.add(self.spike_counter,self.vth),self.spike_counter_int)

        if self.conf.f_record_first_spike_time:
#            self.first_spike_time = tf.where(
#                                        self.f_fire,\
#                                #tf.where(tf.equal(self.first_spike_time,tf.constant(-1,shape=self.first_spike_time.shape)),\
#                                        tf.where(
#                                            self.first_spike_time==tf.constant(-1,shape=self.first_spike_time.shape),\
#                                            tf.constant(t,dtype=tf.float32,shape=self.first_spike_time.shape),\
#                                            self.first_spike_time),\
#                                        self.first_spike_time)

            self.first_spike_time = tf.where(
                                        self.f_fire,\
                                        tf.constant(t,dtype=tf.float32,shape=self.first_spike_time.shape),\
                                        self.first_spike_time)






    # run fwd
    def run_type_in(self,inputs,t):
        #print('run_type_in')
        self.input_spike_gen(inputs,t)
        self.count_spike(t)

    #
    def run_type_if(self,inputs,t):
        #print('run_type_if')

#        f_run_int = t >= (self.depth-1)*self.conf.time_window and t < (self.depth)*self.conf.time_window
#
#        # intergation
#        #{
#        #    'TEMPORAL': self.integration_temporal if f_run else lambda inputs, t : None
#        #}.get(self.neural_coding, self.integration) (inputs,t)
#

        # run_fire flag of each layer
        #f_run_fire = t >= self.depth*self.conf.time_fire_start  and t < (self.depth+1)*self.conf.time_window

        t_fire_s = self.depth*self.conf.time_fire_start
        t_fire_e = t_fire_s + self.conf.time_fire_duration

        f_run_fire = t >= t_fire_s and t < t_fire_e

        #f_run_fire = t >= self.depth*self.conf.time_window and t < (self.depth+1)*self.conf.time_window
        #f_run_fire = t >= self.depth*self.conf.time_window

        # integration
        self.integration(inputs,t)

        # fire
        fire={
            'RATE': self.fire,
            'WEIGHTED_SPIKE': self.fire_weighted_spike,
            'BURST': self.fire_burst,
            #'TEMPORAL': self.fire_temporal if t >= self.depth*self.conf.time_window and t < (self.depth+1)*self.conf.time_window else self.spike_dummy_fire
            #'TEMPORAL': self.fire_temporal if t >= self.depth*self.conf.time_window and t < (self.depth+1)*self.conf.time_window else self.spike_dummy_fire
            'TEMPORAL': self.fire_temporal if f_run_fire else self.spike_dummy_fire
        }

        fire[self.neural_coding](t)

        self.count_spike(t)

    #
    def run_type_lif(self,inputs,t):
        #print('run_type_lif')
        self.leak()
        self.integration(inputs,t)
        self.fire(t)
        self.count_spike(t)

    def run_type_out(self,inputs,t):
        #print("output layer")
        self.integration(inputs,t)
        #self.fire_type_out(t)

    def set_vth(self,vth):
        #self.vth = self.vth.assign(vth)
        self.vth.assign(vth)

    def get_spike_count(self):
        spike_count = tf.reshape(self.spike_counter,self.dim)
        return spike_count

    def get_spike_count_int(self):
        #spike_count_int = tf.reshape(self.spike_counter_int,self.dim)
        return self.spike_counter_int

    def get_spike_rate(self):
        #return self.get_spike_count_int()/self.conf.time_step
        return self.get_spike_count_int()/self.conf.time_step

    def get_tot_psp(self):
        return self.tot_psp

    def get_isi(self):
        return self.isi



    def set_time_const_init_integ(self, time_const_init_integ):
        self.time_const_init_integ = time_const_init_integ

    def set_time_const_init_fire(self, time_const_init_fire):
        self.time_const_init_fire = time_const_init_fire

    def set_time_delay_init_integ(self, time_delay_init_integ):
        self.time_delay_init_integ = time_delay_init_integ

    def set_time_delay_init_fire(self, time_delay_init_fire):
        self.time_delay_init_fire = time_delay_init_fire



    def set_time_const_integ(self, time_const_integ):
        self.time_const_integ = time_const_integ

    def set_time_delay_integ(self, time_delay_integ):
        self.time_delay_integ = time_delay_integ

    # training time constant (fire) for temporal coding
    def train_time_const_fire(self,dnn_act):
        #print("snn_lib: train_time_const")
        #print(dnn_act)
        #self.time_const_integ = tf.zeros([])
        #self.time_const_fire = tf.multiply(self.time_const_fire,0.1)

        # delta - -1/(2tau^2)(x-x_hat)(x_hat)

        spike_time = self.first_spike_time-self.depth*self.conf.time_fire_start-self.time_delay_fire
        x = dnn_act
        x_hat = tf.where(
                    tf.equal(self.first_spike_time,tf.constant(self.init_first_spike_time,shape=self.first_spike_time.shape,dtype=tf.float32)), \
                    tf.zeros(self.first_spike_time.shape), \
                    tf.exp(-(spike_time/self.time_const_fire)))


        #x_hat = tf.exp(-self.first_spike_time/self.time_const_fire)

        #loss = tf.reduce_sum(tf.square(x-x_hat))


        loss = tf.reduce_mean(tf.square(x-x_hat))

        # l2
        delta1 = tf.subtract(x,x_hat)
        delta1 = tf.multiply(delta1,x_hat)
        delta1 = tf.multiply(delta1, tf.subtract(self.first_spike_time,self.time_delay_fire))

        if tf.equal(tf.size(tf.boolean_mask(delta1,delta1>0)),0):
            delta1 = tf.zeros([])
        else:
            delta1 = tf.reduce_mean(tf.boolean_mask(delta1,delta1>0))


        if self.conf.f_train_time_const_outlier:
            #x_min = tf.tfp.stats.percentile(tf.boolean_mask(x,x>0),0.01)
            x_min = tf.constant(np.percentile(tf.boolean_mask(x,x>0).numpy(),0.01),dtype=tf.float32,shape=[])

            #print("min: {:e}, min_0.01: {:e}".format(tf.reduce_min(tf.boolean_mask(x,x>0)),x_min))
        else:
            x_min = tf.reduce_min(tf.boolean_mask(x,x>0))

        x_hat_min = tf.exp(-(self.conf.time_fire_duration/self.time_const_fire))
        delta2 = tf.subtract(x_min,x_hat_min)
        delta2 = tf.multiply(delta2,x_hat_min)
        delta2 = tf.multiply(delta2, tf.subtract(self.conf.time_window,self.time_delay_fire))

        #
        #idx=0,0,0,0
        #print("x: {:e}, x_hat: {:e}".format(x[idx],x_hat[idx]))
        #print("x_min: {:e}, x_hat_min: {:e}".format(x_min,x_hat_min))

        # l1


        #
        delta1 = tf.divide(delta1,tf.square(self.time_const_fire))
        delta2 = tf.divide(delta2,tf.square(self.time_const_fire))


        rho1 = 1000.0
        rho2 = 1000.0

        #
        delta = tf.add(tf.multiply(delta1,rho1),tf.multiply(delta2,rho2))

        #print("name: {:s}, del: {:e}, del1: {:e}, del2: {:e}".format(self.n_name,delta,delta1,delta2))


        #self.time_const_fire = tf.subtract(self.time_const_fire, delta)
        self.time_const_fire = tf.add(self.time_const_fire, delta)


        #
        #idx=0,10,10,0
        #print('x: {:e}, vmem: {:e}, x_hat: {:e}, delta: {:e}'.format(x[idx],self.vmem[idx],x_hat[idx],delta))

        #print("name: {:s}, loss: {:f}, tc: {:f}".format(self.n_name,loss,self.time_const_fire))
        print("name: {:s}, tc: {:f}".format(self.n_name,self.time_const_fire))

        #print("\n")

    def train_time_delay_fire(self, dnn_act):

        if self.conf.f_train_time_const_outlier:
            #x_min = tf.tfp.stats.percentile(tf.boolean_mask(x,x>0),0.01)
            x_max = tf.constant(np.percentile(dnn_act.numpy(),99.9),dtype=tf.float32,shape=[])

            #print("max: {:e}, max_99.9: {:e}".format(tf.reduce_max(dnn_act),x_max))
        else:
            x_max = tf.reduce_max(dnn_act)

        x_max_hat = tf.exp(self.time_delay_fire/self.time_const_fire)

        delta = tf.subtract(x_max,x_max_hat)
        delta = tf.multiply(delta,x_max_hat)
        delta = tf.divide(delta,self.time_const_fire)

        rho = 10.0

        delta = tf.multiply(delta,rho)

        #self.time_delay_fire = tf.subtract(self.time_delay_fire,delta)
        self.time_delay_fire = tf.add(self.time_delay_fire,delta)

        #print("name: {:s}, del: {:e}, td: {:e}".format(self.n_name,delta,self.time_delay_fire))
        print("name: {:s}, td: {:f}".format(self.n_name,self.time_delay_fire))


def spike_max_pool(feature_map, spike_count, output_shape):
    #tmp = tf.reshape(spike_count,(1,-1,)+spike_count.numpy().shape[2:])
    tmp = tf.reshape(spike_count,(1,-1,)+tuple(tf.shape(spike_count)[2:]))
    _, arg = tf.nn.max_pool_with_argmax(tmp,(1,2,2,1),(1,2,2,1),padding='SAME')
    #arg = tf.reshape(arg,output_shape)
    conv_f = tf.reshape(feature_map,[-1])
    arg = tf.reshape(arg,[-1])

    p_conv = tf.gather(conv_f, arg)
    p_conv = tf.reshape(p_conv,output_shape)

    #p_conv = tf.convert_to_tensor(conv_f.numpy()[arg],dtype=tf.float32)

    return p_conv


#def spike_max_pool_temporal(feature_map, spike_count, output_shape):


#def spike_max_pool(feature_map, spike_count, output_shape):
#
#    max_pool = {
#        'TEMPORAL': spike_max_pool_rate
#    }.get(self.neural_coding, spike_max_pool_rate)
#
#    p_conv = max_pool(feature_map, spike_count, output_shape)
#
##    return p_conv
