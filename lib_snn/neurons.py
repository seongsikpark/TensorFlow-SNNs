
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=np.inf)


import tensorflow as tf
from keras import backend

from lib_snn.sim import glb

#
# class Neuron(tf.layers.Layer):
class Neuron(tf.keras.layers.Layer):
    def __init__(self ,dim ,conf ,n_type ,neural_coding ,depth=0 ,name='' ,**kwargs):

        # def __init__(self, dim, n_type, fan_in, conf, neural_coding, depth=0, name='', **kwargs):
        # super(Neuron, self).__init__(name="")
        super(Neuron, self).__init__(name=name, **kwargs)

        self.init_done = False

        self.dim = dim
        #self.dim_one_batch = [1 , ] +dim[1:]

        self.n_type = n_type
        # self.fan_in = fan_in

        self.conf = conf

        self.neural_coding =neural_coding

        #
        #self.vmem = None

        # stat for weighted spike
        # self.en_stat_ws = True
        self.en_stat_ws = False

        #
        self.zeros = tf.zeros(self.dim, dtype=tf.float32)
        #self.fires = tf.constant(self.conf.n_init_vth, shape=self.dim ,dtype=tf.float32)
        self.fires = tf.ones(self.dim, dtype=tf.float32)

        self.depth = depth

        #
        self.num_neurons = tf.cast(tf.divide(tf.reduce_prod(self.dim),self.dim[0]),dtype=tf.float32)

        # if self.conf.f_record_first_spike_time:
        #self.init_first_spike_time = -1
        self.init_first_spike_time = np.nan
        #self.init_first_spike_time = -1.0
        # self.init_first_spike_time = self.conf.time_fire_duration*self.conf.init_first_spike_time_n
        #self.init_first_spike_time = 100000

        #
        #leak_const = -0.01/16*depth+1.0
        #leak_const = -0.001/16*depth+1.0
        #leak_const = -0.009/16*depth+0.999

        #leak_const = -0.009/16*depth+0.99

        #leak_const = -0.009/16*depth+0.98
        #leak_const = -0.001/16*depth+0.98
        #leak_const = -0.009/16*depth+1.0
        #leak_const = 1.0
        #leak_const = 0.99
        #leak_const = 0.96
        #leak_const = 0.95
        #leak_const = 0.94
        #leak_const = 0.93
        #leak_const = 0.92
        #leak_const = 0.91
        #leak_const = 0.9
        #leak_const = 0.89
        #leak_const = 0.88
        #leak_const = 0.87
        #leak_const = 0.86
        #leak_const = 0.85
        #leak_const = 0.84
        #leak_const = 0.83
        #leak_const = 0.82
        #leak_const = 0.81
        #leak_const = 1.0

        #if depth > 14 :
            #leak_const = -0.16 / 15 / 15 * (depth) * (depth) + 1.0
        #else:
            #leak_const = 1.0


        #leak_const = 1.0
        #leak_const = 0.99
        #leak_const = 0.987
        leak_const = 0.985
        #leak_const = -0.006*self.depth+0.991
        #leak_const = -0.003*self.depth+0.991
        #leak_const = -0.0001*self.depth+0.99
        #leak_const = 0.983
        #leak_const = 0.98

        #if depth==4:
            #leak_const = 0.99

        #if depth==15:
        if False:
            leak_const = 0.98
            #leak_const = 0.97
            #leak_const = 0.99



        #if depth==16:
        if self.n_type=='OUT':
            #leak_const = 0.8
            leak_const = 0.85
            #leak_const = 0.87
            #leak_const = 0.90
            #leak_const = 0.91
            #leak_const = 0.94
            #leak_const = 0.95      # VGG

            #leak_const = 0.90
            #leak_const = 0.96
            #leak_const = 0.97
            #leak_const = 0.98
            #leak_const = 0.99
            #leak_const = 0.92
            #leak_const = 0.93
            #leak_const = 0.97
            #leak_const = 0.98
            #leak_const = 0.985
            #leak_const = 0.99
            #leak_const = 0.1


        #leak_const = 0.7
        #leak_const = 0.8
        #self.leak_const = tf.constant(0.99,dtype=tf.float32,shape=self.dim)
        #self.leak_const = tf.constant(leak_const,dtype=tf.float32,shape=self.dim)

        self.leak_const_init = tf.constant(leak_const,dtype=tf.float32,shape=self.dim,name='leak_const_init')
        #self.leak_const = tf.Variable(self.leak_const_init,dtype=tf.float32,shape=self.dim,name='leak_const')
        self.leak_const = tf.Variable(self.leak_const_init,trainable=False,dtype=tf.float32,shape=self.dim,name='leak_const')


        # vth scheduling
        self.vth_schedule = []


        if self.conf.neural_coding =='TEMPORAL':
            assert False
            self.time_const_init_fire = self.conf.tc
            self.time_const_init_integ = self.conf.tc

            self.time_delay_init_fire = 0.0
            self.time_delay_init_integ = 0.0


            if self.conf.f_tc_based:
                self.time_start_integ_init = (self.depth -1 ) *self.conf.time_fire_start
                self.time_start_fire_init = (self.depth ) *self.conf.time_fire_start

                self.time_end_integ_init = self.time_start_integ_init + self.conf.time_fire_duration
                self.time_end_fire_init = self.time_start_fire_init + self.conf.time_fire_duration
            else:
                self.time_start_integ_init = (self.depth -1 ) *self.conf.time_fire_start
                self.time_start_fire_init = (self.depth ) *self.conf.time_fire_start

                self.time_end_integ_init = self.time_start_integ_init + self.conf.time_fire_duration
                self.time_end_fire_init = self.time_start_fire_init + self.conf.time_fire_duration

        # self.spike_count = tf.Variable(name="spike_count",dtype=tf.float32,initial_value=tf.zeros(self.dim,dtype=tf.float32),trainable=False)
        # self.spike_count_int = tf.Variable(name="spike_count_int",dtype=tf.float32,initial_value=tf.zeros(self.dim,dtype=tf.float32),trainable=False)
        # self.f_fire = tf.Variable(name='f_fire', dtype=tf.bool, initial_value=tf.constant(False,dtype=tf.bool,shape=self.dim),trainable=False)

        #
        # self.vmem = tf.Variable(shape=self.dim,dtype=tf.float32,initial_value=tf.constant(self.conf.n_init_vinit,shape=self.dim),trainable=False,name='vmem')

        # SNN training
        self.grad_in_prev = tf.Variable(initial_value=tf.zeros(self.dim), trainable=False, name='grad_in_prev')

        # TODO: conditional - SNN direct training, test method
        self.dL_du_t1_prev = tf.Variable(initial_value=tf.zeros(self.dim),trainable=False,name='dL_du_t1_prev')
        #self.dL_du_t1_prev = None

    def build(self, input_shapes):
        #print('neuron build - {}'.format(self.name))
        super().build(input_shapes)


        if self.n_type == 'IN':
            init_vth = self.conf.n_in_init_vth
        else:
            init_vth = self.conf.n_init_vth

        #self.vth_init = self.add_variable("vth_init" ,shape=self.dim ,dtype=tf.float32 ,initializer=tf.constant_initializer(init_vth) ,trainable=False)
        self.vth_init = tf.constant(init_vth,shape=self.dim,dtype=tf.float32, name='vth_init')

        # self.vth_init = tfe.Variable(init_vth)
        self.vth = tf.Variable(initial_value=tf.constant(init_vth,dtype=tf.float32,shape=self.dim), trainable=False, name="vth")
        #self.vth = tf.constant(init_vth,dtype=tf.float32,shape=self.dim, name="vth")

        self.vmem_init = tf.Variable(initial_value=tf.constant(self.conf.n_init_vinit,dtype=tf.float32,shape=self.dim), trainable=False,name='vmem_init')
        self.vmem = tf.Variable(initial_value=tf.constant(self.conf.n_init_vinit,dtype=tf.float32,shape=self.dim), trainable=False,name='vmem')

        # self.vmem = tf.Variable("vmem",shape=self.dim,dtype=tf.float32,initial_value=tf.constant(self.conf.n_init_vinit),trainable=False)
        # self.vmem = tf.Variable(shape=self.dim,dtype=tf.float32,initial_value=tf.constant(self.conf.n_init_vinit,shape=self.dim),trainable=False,name='vmem')

        #self.out = tf.Variable(initial_value=tf.zeros(self.dim, dtype=tf.float32), trainable=False, name="out")
        self.out = tf.zeros(self.dim, dtype=tf.float32, name="out")

        #
        #self.fire_vmem_sub = tf.constant(init_vth,shape=self.dim,dtype=tf.float32, name='fire_vmem_sub')     # reset by subtract amount

        # relative spike time of each layer
        #self.first_spike_time = tf.Variable(initial_value=tf.constant(self.init_first_spike_time,dtype=tf.float32,shape=self.dim), trainable=False,name='first_spike_time')
        self.first_spike_time = tf.constant(self.init_first_spike_time,dtype=tf.float32,shape=self.dim, name='first_spike_time')


        self.spike_count_int = tf.Variable(initial_value=tf.zeros(self.dim, dtype=tf.float32), trainable=False, name="spike_count_int")
        self.spike_count = tf.Variable(initial_value=tf.zeros(self.dim, dtype=tf.float32), trainable=False, name="spike_count")


        self.f_fire = tf.Variable(initial_value=tf.constant(False,dtype=tf.bool,shape=self.dim), trainable=False, name="f_fire")
        # shape=self.dim, dtype=tf.bool, trainable=False, name="f_fire")



        #
        if self.conf.f_isi:
            assert False
            self.last_spike_time = self.add_variable("last_spike_time" ,shape=self.dim ,dtype=tf.float32
                                                     ,initializer=tf.zeros_initializer ,trainable=False)
            self.isi = self.add_variable("isi" ,shape=self.dim ,dtype=tf.float32 ,initializer=tf.zeros_initializer
                                         ,trainable=False)

        if self.conf.f_tot_psp:
            self.tot_psp = self.add_variable("tot_psp" ,shape=self.dim ,dtype=tf.float32
                                             ,initializer=tf.zeros_initializer ,trainable=False)

        if self.conf.f_refractory:
            self.refractory = self.add_variable("refractory" ,shape=self.dim ,dtype=tf.float32
                                                ,initializer=tf.zeros_initializer ,trainable=False)
            self.t_set_refractory= self.add_variable("t_set_refractory" ,shape=self.dim ,dtype=tf.float32
                                                      ,initializer=tf.constant_initializer(-1.0) ,trainable=False)

        # self.depth = self.add_variable("depth",shape=self.dim,dtype=tf.int32,initializer=tf.zeros_initializer,trainable=False)

        # if self.conf.neural_coding=='TEMPORAL':
        # if self.conf.f_record_first_spike_time:
        #    self.first_spike_time=self.add_variable("first_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.init_first_spike_time),trainable=False)


        # stat for weighted spike
        if self.en_stat_ws:
            self.stat_ws = self.add_variable("stat_ws" ,shape=self.conf.p_ws ,dtype=tf.float32
                                             ,initializer=tf.zeros_initializer ,trainable=False)

        if self.conf.neural_coding=='TEMPORAL':
            # if self.conf.f_record_first_spike_time:
            #    self.first_spike_time=self.add_variable("first_spike_time",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.init_first_spike_time),trainable=False)

            # self.time_const=self.add_variable("time_const",shape=self.dim,dtype=tf.float32,initializer=tf.constant_initializer(self.conf.tc),trainable=False)

            # TODO: old - scalar version
            self.time_const_integ =self.add_variable("time_const_integ" ,shape=[] ,dtype=tf.float32
                                                      ,initializer=tf.constant_initializer(self.time_const_init_integ)
                                                      ,trainable=False)
            self.time_const_fire =self.add_variable("time_const_fire" ,shape=[] ,dtype=tf.float32
                                                     ,initializer=tf.constant_initializer(self.time_const_init_fire)
                                                     ,trainable=False)
            self.time_delay_integ =self.add_variable("time_delay_integ" ,shape=[] ,dtype=tf.float32
                                                      ,initializer=tf.constant_initializer(self.time_delay_init_integ)
                                                      ,trainable=False)
            self.time_delay_fire =self.add_variable("time_delay_fire" ,shape=[] ,dtype=tf.float32
                                                     ,initializer=tf.constant_initializer(self.time_delay_init_fire)
                                                     ,trainable=False)

            self.time_start_inte  =self.add_variable("time_start_integ" ,shape=[] ,dtype=tf.float32
                                                      ,initializer=tf.constant_initializer(self.time_start_integ_init)
                                                      ,trainable=False)
            self.time_end_integ =self.add_variable("time_end_integ" ,shape=[] ,dtype=tf.float32
                                                    ,initializer=tf.constant_initializer(self.time_end_integ_init)
                                                    ,trainable=False)
            self.time_start_fire =self.add_variable("time_start_fire" ,shape=[] ,dtype=tf.float32
                                                     ,initializer=tf.constant_initializer(self.time_start_fire_init)
                                                     ,trainable=False)
            self.time_end_fire =self.add_variable("time_end_fire" ,shape=[] ,dtype=tf.float32
                                                   ,initializer=tf.constant_initializer(self.time_end_fire_init)
                                                   ,trainable=False)

            #            self.time_const_integ=self.add_variable("time_const_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_integ),trainable=False)
            #            self.time_const_fire=self.add_variable("time_const_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_const_init_fire),trainable=False)
            #            self.time_delay_integ=self.add_variable("time_delay_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_integ),trainable=False)
            #            self.time_delay_fire=self.add_variable("time_delay_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_delay_init_fire),trainable=False)
            #
            #            self.time_start_integ=self.add_variable("time_start_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_integ_init),trainable=False)
            #            self.time_end_integ=self.add_variable("time_end_integ",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_integ_init),trainable=False)
            #            self.time_start_fire=self.add_variable("time_start_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_start_fire_init),trainable=False)
            #            self.time_end_fire=self.add_variable("time_end_fire",shape=self.dim_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.time_end_fire_init),trainable=False)


            print_loss =True

            if self.conf.f_train_tk and print_loss:
                self.loss_prec =self.add_variable("loss_prec" ,shape=[] ,dtype=tf.float32
                                                   ,initializer=tf.zeros_initializer ,trainable=False)
                self.loss_min =self.add_variable("loss_min" ,shape=[] ,dtype=tf.float32
                                                  ,initializer=tf.zeros_initializer ,trainable=False)
                self.loss_max  =self.add_variable("loss_max" ,shape=[] ,dtype=tf.float32
                                                  ,initializer=tf.zeros_initializer ,trainable=False)


        #        if self.conf.noise_en and (self.conf.noise_type=="JIT"):
        #            #self.jit_max = tf.floor(4*self.conf.noise_pr)       # 4 sigma
        #            self.jit_max = int(tf.floor(4*self.conf.noise_pr))       # 4 sigma
        #            shape = tensor_shape.TensorShape(self.vmem.shape)
        #            #shape = tensor_shape.TensorShape(self.out.shape+[int(self.jit_max),])
        #            #shape = tensor_shape.TensorShape([int(self.jit_max),]+self.out.shape)
        #
        #            #self.jit_q = self.add_variable("jit_q",shape=shape,dtype=tf.bool,initializer=tf.constant_initializer(False),trainable=False)
        #            self.jit_q = self.add_variable("jit_q",shape=shape,dtype=tf.int32,initializer=tf.constant_initializer(False),trainable=False)

        self.built = True

    #@tf.function
    @tf.custom_gradient
    def call(self ,inputs ,t, training):

        #print('neuron call')
        #if self.depth==15:
        #print('depth: {}'.format(self.depth))
        #print(inputs)
        #assert self.init_done, 'should call init() before start simulation'

        self.vmem_pre = self.vmem

        #
        #def grad(upstream, variables=None):
        def grad(upstream):
            #
            #self.dL_du_t1_prev = tf.where(t==0,tf.zeros(upstream.shape),self.dL_du_t1_prev)
            #self.dL_du_t1_prev = tf.where(t==0,tf.zeros(upstream.shape),self.dL_du_t1_prev)


#            if variables is not None:
#                print(variables)
#                assert False
            #print('grad')
            #print(self.name)
            #print(tf.reduce_mean(upstream))

            #if self.name=='n_predictions':
            #    print(upstream)

            if self.n_type=='OUT':
                #grad_ret = upstream/10.0
                #if self.conf.dataset!='CIFAR10':
                #    assert False
                grad_ret = upstream
                #grad_ret = tf.divide(upstream,self.conf.batch_size)
                #print(upstream)
                #grad_ret = tf.divide(upstream,self.conf.time_step)
            else:
                # TODO: parameterize
                a=0.5
                if True:
                #if False:
                    cond_1=tf.math.less_equal(tf.math.abs(self.vmem_pre-self.vth),a)
                    #cond_1 = tf.math.logical_and(cond_1,tf.math.logical_not(self.f_fire))
                    #cond_2 = self.f_fire
                    #cond = tf.math.logical_or(cond_1,cond_2)
                    cond = cond_1
                    do_du = tf.where(cond,tf.ones(cond.shape),tf.zeros(cond.shape))
                else:
                    #cond=tf.math.less_equal(tf.math.abs(self.vth-inputs),a)
                    #cond = tf.math.greater_equal(inputs, tf.zeros(shape=inputs.shape))
                    cond = tf.math.greater_equal(self.vmem, tf.zeros(shape=self.vmem.shape))
                # cond_1 = tf.math.logical_or(tf.greater_equal(self.vmem,(1.0-a)*self.vth),tf.less_equal(self.vmem,(1.0+a)*self.vth))
                # cond = cond_1
                # cond = cond_2
                # spatio = tf.where(cond, upstream, tf.zeros(upstream.shape))  # spaio BP only
                # spatio = tf.multiply(upstream,do_du)  # spaio BP only
                spatio = upstream

                #
                # dL_du_t1 = tf.random.normal(spatio.shape,mean=1.0,stddev=0.1)
                dL_du_t1 = upstream  # speculate next time step gradient - dL/du_t+1
                # test - 221209
                #dL_du_t1 = dL_du_t1 * do_du

                if False:
                    # if tf.cond(hasattr(self, 'dL_du_t1_prev'):
                    # dL_du_t1 = dL_du_t1+self.dL_du_t1_prev
                    # print(self.dL_du_t1_prev == tf.zeros([]))

                    # self.dL_du_t1_prev = tf.where(self.dL_du_t1_prev == tf.zeros([]),
                    #                              tf.Variable(initial_value=tf.zeros(upstream.shape)),
                    #                              self.dL_du_t1_prev)

                    # dL_du_t1 = tf.cond(self.dL_du_t1_prev == tf.zeros([]),lambda: tf.identity(dL_du_t1),
                    #                   lambda: tf.math.add(dL_du_t1,self.dL_du_t1_prev))
                    dL_du_t1 = tf.math.add(dL_du_t1, self.dL_du_t1_prev)
                    dL_du_t1 = dL_du_t1 / tf.cast(self.conf.time_step,
                                                  dtype=tf.float32)
                elif True:
                    # dL_du_t1 = dL_du_t1
                    # dL_du_t1 = dL_du_t1*do_du
                    # dL_du_t1 = tf.cond(tf.equal(self.grad_in_prev,tf.zeros(self.dim)),
                    #                    upstream,
                    #                    upstream - self.grad_in_prev)
                    # dL_du_t1 = (dL_du_t1 - self.grad_in_prev)*do_du

                    dL_du_t1 = (dL_du_t1 + self.grad_in_prev) / 2

                # temp_rand = upstream * temp_rand           # speculate next time step gradient

                # temporal_1 = du_(t+1)/do_(t)
                # temporal_1 = tf.where(self.f_fire, -self.vth, tf.zeros(spatio.shape))
                temporal_1 = tf.where(self.f_fire, -self.vth, tf.zeros(spatio.shape))  # reset-by-sub
                #temporal_1= tf.where(self.f_fire, -self.vmem_pre, tf.zeros(spatio.shape))  # reset-to-zero
                # temporal_1 = tf.zeros(spatio.shape)

                # temporal_2 = tf.ones(spatio.shape)          # du_(t+1)/du_(t) # IF
                temporal_2 = self.leak_const_init  # du_(t+1)/du_(t) # LIF

                dL_do = spatio + dL_du_t1 * temporal_1
                dL_du = dL_do * do_du + dL_du_t1 * temporal_2

                grad_ret = dL_du
                # grad_ret = dL_do*do_du
                # grad_ret = spatio*do_du

                # test
                # grad_ret = dL_do
                # grad_ret = dL_du_t1*temporal_2
                # grad_ret = dL_du_t1
                # grad_ret = spatio*do_du
                # grad_ret = upstream*do_du
                #grad_ret = upstream

                # print()
                ##if self.name=='n_fc1' or self.name=='n_fc2':
                # if True:
                ##print(self.name)
                # print(upstream)
                # print(grad_ret)

                # print(self.name)
                # print(grad_ret)

                # if tf.math.is_nan(tf.reduce_mean(grad_ret)):
                # print(self.name)
                # print(upstream)
                # assert False

                # temporal = tf.add(temporal_1,temporal_2)
                # temporal = tf.multiply(temporal, dL_du_t1)

                # grad_ret = spatio + temporal
                # grad_ret = spatio
                # grad_ret = tf.where(cond,upstream,-upstream)
                # grad_ret = tf.clip_by_norm(grad_ret,0.1)
                # grad_ret = tf.clip_by_norm(grad_ret,2)
                # grad_ret = upstream

                self.grad_in_prev.assign(dL_du_t1)
                # self.grad_in_prev.assign(dL_du_t1)

                # test
                # grad_ret = spatio*do_du

                if self.conf.debug_mode and self.conf.verbose_snn_train:
                    print("grad_ret> {} - max: {:.3g}, min: {:.3g}, mean: {:.3g}, var: {:.3g}"
                          .format(self.name,tf.reduce_max(grad_ret),tf.reduce_min(grad_ret),tf.reduce_mean(grad_ret),tf.math.reduce_variance(grad_ret)))

                # self.dL_du_t1_prev = dL_du_t1
                self.dL_du_t1_prev.assign(dL_du_t1)

            # print('here')
            # print(self.name)
            # print(upstream)
            # print(self.out)

            return grad_ret, tf.stop_gradient(t), tf.stop_gradient(training)


        self.inputs = inputs

        #
        if self.conf.leak_time_dep and (self.n_type!='OUT'):
        #if self.conf.leak_time_dep:
            self.set_leak_time_dep(t)

        # run_fwd
        run_type = {
            'IN': self.run_type_in,
            'IF': self.run_type_if,
            'LIF': self.run_type_lif,
            'OUT': self.run_type_out
        }
        #}[self.n_type](inputs ,t, training)
        #out_ret = self.out

        out = run_type[self.n_type](inputs, t, training)
        out_ret = out

        #self.t = t
        #print(self.conf.time_step)
        #print(t)

        #if True:
        if False:
            print('name: {:}'.format(self.name))
            #print(self.vmem[0,0])
            print(self.out[0,0])

        if False:
        #if True:
            if self.n_type=='OUT':
                #if (self.conf.time_step==t):
                print('snn out_ret')
                print(self.vmem[0])
                print(out_ret[0])
                print('')



        #if self.name=='n_predictions':
        #    print('output')
        #    print(out_ret)

        #
        if self.conf.f_record_first_spike_time:
            self.record_first_spike_time(t)

        #if self.depth==1:
            #print(self.vmem)
            #print(tf.reduce_mean(self.out))


        return out_ret, grad

    # initialization
    def init(self):

        self.init_done = True

    # reset - time step
    def reset_each_time(self):
        self.reset_out()

    # reset - sample
    def reset(self):
        #print('reset neuron')
        self.reset_vmem()
        # self.reset_out()
        self.reset_spike_count()
        self.reset_vth()

        if self.conf.leak_time_dep:
            self.reset_leak_const()

        if self.conf.f_tot_psp:
            assert False
            self.reset_tot_psp()
        if self.conf.f_isi:
            assert False
            self.last_spike_time = tf.zeros(self.last_spike_time.shape)
            self.isi = tf.zeros(self.isi.shape)
        if self.conf.f_refractory:
            assert False
            self.refractory = tf.zeros(self.refractory.shape)
            self.t_set_refractory = tf.constant(-1.0 ,dtype=tf.float32 ,shape=self.refractory.shape)

        if self.conf.f_record_first_spike_time:
            self.reset_first_spike_time()

        # TODO: add condition
        self.reset_snn_direct_training()

    def reset_spike_count(self):
        # self.spike_count = tf.zeros(self.dim)
        # self.spike_count_int = tf.zeros(self.dim)

        self.spike_count.assign(tf.zeros(self.dim ,dtype=tf.float32))
        self.spike_count_int.assign(tf.zeros(self.dim ,dtype=tf.float32))

        # self.spike_count.assign(self.zeros)
        # self.spike_count.assign(tf.zeros(self.dim))
        # self.spike_count_int = tf.zeros(self.out.shape)
        # self.spike_count_int.assign(tf.zeros(self.dim))

    #
    def reset_vmem(self, init_vmem=None):
        # assert False
        # self.vmem = tf.constant(self.conf.n_init_vinit,tf.float32,self.vmem.shape)
        #print(self.name)
        if init_vmem is None:
            #init_vmem = tf.constant(self.conf.n_init_vinit ,tf.float32 ,self.vmem.shape)
            init_vmem = self.vmem_init
        #self.vmem.assign(tf.constant(self.conf.n_init_vinit ,tf.float32 ,self.vmem.shape))

        #f_init_vmem_rand=True
        f_init_vmem_rand=False
        if f_init_vmem_rand:
            init_vmem = tf.random.normal(init_vmem.shape,mean=0.0,stddev=0.5)


        self.vmem.assign(init_vmem)


        # print(type(self.vmem))
        # assert False

    #
    def reset_tot_psp(self):
        self.tot_psp = tf.zeros(tf.shape(self.tot_psp))

    #
    def reset_out(self):
        self.out.assign(tf.zeros(self.out.shape))

    def reset_vth(self):
        self.vth.assign(self.vth_init)
        #self.vth=self.vth_init

    #
    def reset_first_spike_time(self):
        self.first_spike_time =tf.constant(self.init_first_spike_time ,shape=self.first_spike_time.shape
                                            ,dtype=tf.float32)

    #
    def reset_leak_const(self):
        self.leak_const.assign(self.leak_const_init)

    #
    def reset_snn_direct_training(self):
        #self.dL_du_t1_prev = tf.zeros(self.dim)
        #self.dL_du_t1_prev = tf.Variable(initial_value=tf.zeros(self.dim))
        self.dL_du_t1_prev.assign(tf.zeros(self.dim))
        #self.dL_du_t1_prev = None

        #
        self.grad_in_prev.assign(tf.zeros(self.dim))

    #
    def set_vmem_init(self, vmem_init):
        self.vmem_init.assign(vmem_init)

    ##
    def set_vth_temporal_kernel(self ,t):
        # exponential decay
        # self.vth = tf.constant(tf.exp(-float(t)/self.conf.tc),tf.float32,self.out.shape)
        # self.vth = tf.constant(tf.exp(-t/self.conf.tc),tf.float32,self.out.shape)
        time = tf.subtract(t ,self.time_delay_fire)

        if self.conf.f_qvth:
            time = tf.add(time ,0.5)

        # print(self.vth.shape)
        # print(self.time_const_fire.shape)
        # self.vth = tf.constant(tf.exp(tf.divide(-time,self.time_const_fire)),tf.float32,self.out.shape)

        # TODO: check
        self.vth = tf.exp(tf.divide(-time ,self.time_const_fire))
        #
        # self.vth = tf.multiply(self.conf.n_init_vth,tf.exp(tf.divide(-time,self.time_const_fire)))

        # polynomial
        # self.vth = tf.constant(tf.add(-tf.pow(t/self.conf.tc,2),1.0),tf.float32,self.out.shape)

    #
    def set_leak_const(self,leak_const):
        self.leak_const.assign(leak_const)

    #
    def set_leak_time_dep(self, t):

        alpha = 0.99
        #alpha = 0.90
        #if t < 40: # 256 96.32
        if t < 35: # 256 - 96.36
        #if t < 30:  # 256 - 96.35
        #if t < 20:  # 256 - 96.35
            leak_const = self.leak_const_init
        else:
            leak_const = tf.ones(self.leak_const_init.shape)

        #leak_const = self.leak_const_init*(alpha+(1-alpha)/self.conf.time_step*t)


        self.leak_const.assign(leak_const)

    ##
    def input_spike_real(self ,inputs ,t):
        # TODO: check it
        if self.conf.neural_coding=="WEIGHTED_SPIKE":
            self.out =tf.truediv(inputs ,self.conf.p_ws)
        elif self.conf.neural_coding=="TEMPORAL":
            if t== 0:
                self.out = inputs
            else:
                self.out = tf.zeros(self.out.shape)
        else:
            self.out = inputs

    #        else:
    #            self.out=inputs
    #            #assert False
    # self.out=inputs

    def input_spike_poission(self, inputs, t):
        # Poission input
        vrand = tf.random_uniform(self.vmem.shape, minval=0.0, maxval=1.0, dtype=tf.float32)

        self.f_fire = inputs >= vrand

        self.out = tf.where(self.f_fire, self.fires, self.zeros)
        # self.out = tf.where(self.f_fire,tf.constant(1.0,tf.float32,self.out.shape),tf.zeros(self.vmem.shape))

    def input_spike_weighted_spike(self, inputs, t):
        assert False
        # weighted synpase input
        t_mod = (int)(t % 8)
        if t_mod == 0:
            self.vmem = inputs
            self.vth = tf.constant(0.5, tf.float32, self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth, 0.5)

        f_fire = self.vmem >= self.vth

        self.out = tf.where(f_fire, self.vth, tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem, self.out)

    def input_spike_burst(self, inputs, t):
        assert False
        if t == 0:
            self.vmem = inputs

        f_fire = self.vmem >= self.vth

        self.out = tf.where(f_fire, self.vth, tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem, self.out)

        self.vth = tf.where(f_fire, self.vth * 2.0, self.vth_init)

        # repeat input
        if tf.equal(tf.reduce_max(self.out), 0.0):
            self.vmem = inputs

    def input_spike_temporal(self, inputs, t):

        if t == 0:
            self.vmem = inputs

        # kernel = self.temporal_kernel(t)
        # self.vth = tf.constant(kernel,tf.float32,self.out.shape)
        # self.vth = tf.constant(tf.exp(-t/self.conf.tc),tf.float32,self.out.shape)
        # self.vth = self.temporal_kernel(t)
        self.set_vth_temporal_kernel(t)
        # print(self.vth[0,1,1,1])

        # print("input_spike_temporal: depth: "+str(self.depth)+", t_glb: "+str(t)+", t_loc: "+str(t)+", kernel: "+str(self.vth[0,0,0,0].numpy()))

        if self.conf.f_refractory:
            self.f_fire = (self.vmem >= self.vth) & \
                          tf.equal(self.refractory, tf.constant(0.0, tf.float32, self.refractory.shape))
            # (self.vth >= tf.constant(10**(-5),tf.float32,self.vth.shape) ) & \

        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10 ** (-5))

        # print(f_fire)
        # print(self.vth >= 10**(-1))

        # reset by subtraction
        # self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        # self.vmem = tf.subtract(self.vmem,self.out)

        # reset by zero
        self.out = tf.where(self.f_fire, tf.ones(self.out.shape), tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire, tf.zeros(self.out.shape), self.vmem)

        if self.conf.f_refractory:
            # self.cal_refractory_temporal(self.f_fire)
            self.cal_refractory_temporal(t)

        # self.vth = tf.where(f_fire,self.vth*2.0,self.vth_init)

        # repeat input
        # print(tf.reduce_max(self.out))
        # if tf.equal(tf.reduce_max(self.out),0.0):
        #    self.vmem = inputs

    def spike_dummy_input(self, inputs, t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False, tf.bool, self.f_fire.shape)

    def spike_dummy_fire(self, t):
        self.out = tf.zeros(self.out.shape)
        self.f_fire = tf.constant(False, tf.bool, self.f_fire.shape)

    # (min,max) = (0.0,1.0)
    def input_spike_gen(self, inputs, t):

        if self.conf.f_tc_based:
            f_run_temporal = (t < self.conf.n_tau_fire_duration * self.time_const_fire)
        else:
            f_run_temporal = (t < self.conf.time_fire_duration)

        input_spike_mode = {
            'REAL': self.input_spike_real,
            'POISSON': self.input_spike_poission,
            'WEIGHTED_SPIKE': self.input_spike_weighted_spike,
            'BURST': self.input_spike_burst,
            'TEMPORAL': self.input_spike_temporal if f_run_temporal else self.spike_dummy_input
        }

        input_spike_mode[self.conf.input_spike_mode](inputs, t)

    #
    def leak(self):
        #assert False
        self.vmem.assign(tf.multiply(self.vmem, self.leak_const))

    #
    def cal_isi(self, f_fire, t):
        f_1st_spike = self.last_spike_time == 0.0
        f_isi_update = np.logical_and(f_fire, np.logical_not(f_1st_spike))

        self.isi = tf.where(f_isi_update, tf.constant(t, tf.float32, self.isi.shape) - self.last_spike_time,
                            np.zeros(self.isi.shape))

        self.last_spike_time = tf.where(f_fire, tf.constant(t, tf.float32, self.last_spike_time.shape),
                                        self.last_spike_time)

    #
    def integration(self, inputs, t):

        if self.conf.neural_coding == "TEMPORAL":
            t_int_s = self.time_start_integ_init
            t_int_e = self.time_end_integ_init

            f_run_int_temporal = (t >= t_int_s and t < t_int_e) or (t == 0)  # t==0 : for bias integration
        else:
            f_run_int_temporal = False

        # print(type(self.vmem))
        # assert False

        # intergation
        {
            'TEMPORAL': self.integration_temporal if f_run_int_temporal else lambda inputs, t: None,
            'NON_LINEAR': self.integration_non_lin
        }.get(self.neural_coding, self.integration_default)(inputs, t)

        ########################################
        # common
        ########################################

        #
        if self.conf.f_positive_vmem:
            #self.vmem = tf.maximum(self.vmem, tf.constant(0.0, tf.float32, self.vmem.shape))
            self.vmem.assign(tf.maximum(self.vmem, tf.constant(0.0, tf.float32, self.vmem.shape)))

        #
        if self.conf.f_neg_cap_vmem and (not self.n_type=='OUT'):
            #self.vmem.assign(tf.maximum(self.vmem, -self.vth*2))
            self.vmem.assign(tf.maximum(self.vmem, -self.vth))


        #
        if self.conf.f_tot_psp:
            self.tot_psp = tf.add(self.tot_psp, inputs)

        # debug
        # if self.depth==3:
        #    print(self.vmem.numpy())

    #@tf.function
    def integration_default(self, inputs, t):
        self.vmem.assign(tf.add(self.vmem, inputs))

    #
    def integration_temporal(self, inputs, t):
        if (t == 0):
            time = 0
        else:
            time = self.relative_time_integ(t)

            time = time - self.time_delay_integ
        # time = tf.zeros(self.vmem.shape)

        if self.conf.noise_en:
            if self.conf.noise_type == "JIT" or self.conf.noise_type == "JIT-A":
                rand = tf.random.normal(shape=inputs.shape, mean=0.0, stddev=self.conf.noise_pr)

                if self.conf.noise_type == "JIT-A":
                    rand = tf.abs(rand)
                    rand = tf.floor(rand)
                else:
                    f_positive = tf.greater_equal(rand, tf.zeros(shape=rand.shape))
                    rand = tf.where(f_positive, tf.floor(rand), tf.math.ceil(rand))

                time = tf.add(time, rand)

        # receptive_kernel = tf.constant(tf.exp(-time/self.conf.tc),tf.float32,self.vmem.shape)

        # receptive_kernel = tf.constant(tf.exp(-time/self.time_const_integ),tf.float32,self.vmem.shape)
        receptive_kernel = tf.exp(tf.divide(-time, self.time_const_integ))

        # print("integration_temporal: depth: "+str(self.depth)+", t_glb: "+str(t)+", t_loc: "+str(time)+", kernel: "+str(receptive_kernel[0,0,0,0].numpy()))

        #
        # print('test')
        # print(inputs.shape)
        # print(receptive_kernel.shape)

        if self.depth == 1:
            if self.conf.input_spike_mode == 'REAL':
                psp = inputs
            else:
                psp = tf.multiply(inputs, receptive_kernel)
        else:
            psp = tf.multiply(inputs, receptive_kernel)
        # print(inputs[0,0,0,1])
        # print(psp[0,0,0,1])

        # addr=0,10,10,0
        # print("int: glb {}: loc {} - in {:0.3f}, kernel {:.03f}, psp {:0.3f}".format(t, time,inputs[addr],receptive_kernel[addr],psp[addr]))

        #
        # self.vmem = tf.add(self.vmem,inputs)
        self.vmem = tf.add(self.vmem, psp)

    def integration_non_lin(self, inputs, t):

        #
        # alpha=tf.constant(1.0,tf.float32,self.vmem.shape)
        # beta=tf.constant(0.1,tf.float32,self.vmem.shape)
        # gamma=tf.constant(1.0,tf.float32,self.vmem.shape)
        # eps=tf.constant(0.01,tf.float32,self.vmem.shape)
        # non_lin=tf.multiply(alpha,tf.log(tf.add(beta,tf.divide(gamma,tf.abs(tf.add(self.vmem,eps))))))

        #
        # alpha=tf.constant(0.0459,tf.float32,self.vmem.shape)
        # beta=tf.constant(-7.002,tf.float32,self.vmem.shape)

        #
        alpha = tf.constant(0.3, tf.float32, self.vmem.shape)
        beta = tf.constant(-2.30259, tf.float32, self.vmem.shape)

        # alpha=tf.constant(0.459,tf.float32,self.vmem.shape)
        # alpha=tf.constant(1.5,tf.float32,self.vmem.shape)

        non_lin = tf.multiply(alpha, tf.exp(tf.multiply(beta, tf.abs(self.vmem))))

        psp = tf.multiply(inputs, non_lin)

        dim = tf.size(tf.shape(inputs))
        if tf.equal(dim, tf.constant(4)):
            idx = 0, 0, 0, 0
        elif tf.equal(dim, tf.constant(3)):
            idx = 0, 0, 0
        elif tf.equal(dim, tf.constant(2)):
            idx = 0, 0

        # print("vmem: {:g}, non_lin: {:g}, inputs: {:g}, psp: {:g}".format(self.vmem[idx],non_lin[idx],inputs[idx],psp[idx]))

        self.vmem = tf.add(self.vmem, psp)

    # TODO: move
    ############################################################
    ## noise function
    ############################################################
    def noise_jit(self, t):
        rand = tf.random.normal(self.out.shape, mean=0.0, stddev=self.conf.noise_pr)
        time_jit = tf.cast(tf.floor(tf.abs(rand)), dtype=tf.int32)

        f_jit_del = tf.not_equal(time_jit, tf.zeros(self.out.shape, dtype=tf.int32))
        t_mod = t % self.jit_max
        pow_t_mod = tf.pow(2, t_mod)
        one_or_zero = tf.cast(tf.math.mod(tf.truediv(self.jit_q, pow_t_mod, 2), 2), tf.float32)
        f_jit_ins = tf.equal(one_or_zero, tf.ones(self.out.shape))
        self.jit_q = tf.where(f_jit_ins, tf.subtract(self.jit_q, pow_t_mod), self.jit_q)

        f_fire_and_jit_del = tf.math.logical_and(self.f_fire, f_jit_del)

        jit_t = tf.math.floormod(tf.add(tf.constant(t, shape=time_jit.shape, dtype=tf.int32), time_jit), self.jit_max)

        jit_q_t = tf.add(self.jit_q, tf.pow(2, jit_t))

        self.jit_q = tf.where(f_fire_and_jit_del, jit_q_t, self.jit_q)

        # jitter - del
        out_tmp = tf.where(f_fire_and_jit_del, tf.zeros(self.out.shape), self.out)

        # jitter - insert
        # reset by subtraction
        self.out = tf.where(f_jit_ins, self.vth, out_tmp)

    def noise_del(self):
        # print(self.out)
        # print(tf.reduce_sum(self.out))

        rand = tf.random.uniform(self.out.shape, minval=0.0, maxval=1.0)

        f_noise = tf.less(rand, tf.constant(self.conf.noise_pr, shape=self.out.shape))

        f_fire_and_noise = tf.math.logical_and(self.f_fire, f_noise)

        # out_noise = tf.where(f_fire_and_noise,tf.zeros(self.out.shape),self.out)

        # if self.neural_coding=="TEMPORAL":
        # self.out = tf.where(f_fire_and_noise,tf.zeros(self.out.shape),self.out)
        # self.f_fire = tf.where(f_fire_and_noise,tf.constant(False,tf.bool,self.f_fire.shape),self.f_fire)
        # else:
        self.out = tf.where(f_fire_and_noise, tf.zeros(self.out.shape), self.out)
        self.f_fire = tf.where(f_fire_and_noise, tf.constant(False, tf.bool, self.f_fire.shape), self.f_fire)

        # print("out_noise: {}".format(tf.reduce_sum(out_noise)))

        # assert False

    ############################################################
    ## fire function
    ############################################################
    def fire(self, t):

        #
        # for TTFS coding
        #
        if self.conf.neural_coding == "TEMPORAL":
            t_fire_s = self.time_start_fire_init
            t_fire_e = self.time_end_fire_init
            f_run_fire = t >= t_fire_s and t < t_fire_e
        else:
            f_run_fire = False

        {
            'RATE': self.fire_rate,
            'WEIGHTED_SPIKE': self.fire_weighted_spike,
            'BURST': self.fire_burst,
            # 'TEMPORAL': self.fire_temporal if t >= self.depth*self.conf.time_window and t < (self.depth+1)*self.conf.time_window else self.spike_dummy_fire
            'TEMPORAL': self.fire_temporal if f_run_fire else self.spike_dummy_fire,
            'NON_LINEAR': self.fire_non_lin
        }.get(self.neural_coding, self.fire_rate)(t)


        #
        if self.conf.noise_en:
            ## noise - jitter
            # if self.conf.noise_type=='JIT':
            #    self.noise_jit(t)

            # TODO: modify it to switch style
            # noise - DEL spikes
            if self.conf.noise_type == 'DEL':
                self.noise_del()

    #
    # TODO
    # @tf.function
    def fire_condition_check(self):
        return tf.math.greater_equal(self.vmem, self.vth)

    #
    def fire_rate(self, t):
        # self.f_fire = self.vmem >= self.vth
        self.f_fire = self.fire_condition_check()

        if self.conf.binary_spike:
            self.out = tf.where(self.f_fire, self.fires, self.zeros)
        else:
            self.out = tf.where(self.f_fire, self.vth, self.zeros)

        # reset
        # vmem -> vrest

        # TODO: functionalize
        # reset by subtraction
        if self.conf.n_reset_type=='reset_by_sub':

            # self.vmem = tf.where(f_fire,self.vmem-self.vth,self.vmem)
            #self.vmem.assign(tf.subtract(self.vmem, self.out))             # subtract by output
            #self.vmem.assign(tf.subtract(self.vmem, self.vth))
            #self.vmem.assign(tf.where(self.f_fire, tf.subtract(self.vmem,self.fire_vmem_sub),self.vmem))    # subtract by vth or others?

            # vth random
            #vth_stoch=True
            vth_stoch=False
            r = 0.01
            if vth_stoch:
                vth = self.vth + tf.random.uniform(self.vth.shape,minval=0,maxval=self.vth*r*2) - self.vth*r
            else:
                vth = self.vth

            #self.vmem.assign(tf.where(self.f_fire, tf.subtract(self.vmem,self.vth),self.vmem))    # subtract by vth or others?
            #self.vmem.assign(tf.where(self.f_fire, tf.subtract(self.vmem,vth),self.vmem))    # subtract by vth or others?
            self.vmem.assign(tf.where(self.f_fire, tf.subtract(self.vmem,self.out),self.vmem))    # subtract by vth or others?

        # reset to zero
        elif self.conf.n_reset_type=='reset_to_zero':
            self.vmem.assign(tf.where(self.f_fire,tf.constant(self.conf.n_init_vrest,tf.float32,self.vmem.shape),self.vmem))
        else:
            assert False

        if self.conf.f_isi:
            self.cal_isi(self.f_fire, t)


    def fire_weighted_spike(self, t):
        # weighted synpase input
        t_mod = (int)(t % self.conf.p_ws)
        if t_mod == 0:
            # TODO: check
            # self.vth = tf.constant(0.5,tf.float32,self.vth.shape)
            self.vth = tf.constant(self.conf.n_init_vth, tf.float32, self.vth.shape)
        else:
            self.vth = tf.multiply(self.vth, 0.5)

        if self.conf.f_refractory:
            # self.f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
            # TODO: check
            # self.f_fire = tf.logical_and(self.vmem >= self.vth, tf.equal(self.refractory,0.0))
            assert False, 'modify refractory'
        else:
            self.f_fire = self.vmem >= self.vth

        if self.conf.f_refractory:
            print('fire_weighted_spike, refractory: not implemented yet')

        self.out = tf.where(self.f_fire, self.vth, tf.zeros(self.out.shape))

        self.vmem = tf.subtract(self.vmem, self.out)

        # noise - jit
        if self.conf.noise_en:
            if self.conf.noise_type == "JIT" or self.conf.noise_type == "JIT-A":
                rand = tf.random.normal(shape=self.out.shape, mean=0.0, stddev=self.conf.noise_pr)

                if self.conf.noise_type == "JIT-A":
                    rand = tf.abs(rand)
                    rand = tf.floor(rand)
                else:
                    f_positive = tf.greater_equal(rand, tf.zeros(shape=rand.shape))
                    rand = tf.where(f_positive, tf.floor(rand), tf.math.ceil(rand))

                pow_rand = tf.pow(2.0, -rand)
                out_jit = tf.multiply(self.out, pow_rand)
                self.out = tf.where(self.f_fire, out_jit, self.out)

        # stat for weighted spike
        if self.en_stat_ws:
            count = tf.cast(tf.math.count_nonzero(self.out), tf.float32)
            # print(count)
            # tf.tensor_scatter_nd_add(self.stat_ws,[[t_mod]],[count])
            # self.stat_ws.scatter_add(tf.IndexedSlices(10.0,1))
            self.stat_ws.scatter_add(tf.IndexedSlices(count, t_mod))
            # tf.tensor_scatter_nd_add(self.stat_ws,[[1]],[10])

            print(self.stat_ws)
            # plt.hist(self.stat_ws.numpy())
            # plt.show()

        if self.conf.f_isi:
            self.cal_isi(self.f_fire, t)

    #
    def fire_burst(self, t):
        if self.conf.f_refractory:
            # self.f_fire = np.logical_and(self.vmem >= self.vth,np.equal(self.refractory,0.0))
            # TODO: check
            # self.f_fire = tf.logical_and(self.vmem >= self.vth, np.equal(self.refractory,0.0))
            assert False, 'modify refractory'

        else:
            self.f_fire = self.vmem >= self.vth

        # print(f_fire)
        # print(np.equal(self.refractory,0.0))
        # print(self.refractory)

        # reset by subtraction
        self.out = tf.where(self.f_fire, self.vth, tf.zeros(self.out.shape))
        self.vmem = tf.subtract(self.vmem, self.out)

        if self.conf.f_refractory:
            self.cal_refractory(self.f_fire)

        # exp increasing order
        self.vth = tf.where(self.f_fire, self.vth * 2.0, self.vth_init)
        # self.vth = tf.where(f_fire,self.vth*1.5,self.vth_init)
        # exp decreasing order
        # self.vth = tf.where(f_fire,self.vth*0.5,self.vth_init)
        # self.vth = tf.where(f_fire,self.vth*0.9,self.vth_init)

        if self.conf.noise_en:
            if self.conf.noise_type == "JIT" or self.conf.noise_type == "JIT-A":
                rand = tf.random.normal(shape=self.out.shape, mean=0.0, stddev=self.conf.noise_pr)

                if self.conf.noise_type == "JIT-A":
                    rand = tf.abs(rand)
                    rand = tf.floor(rand)
                else:
                    f_positive = tf.greater_equal(rand, tf.zeros(shape=rand.shape))
                    rand = tf.where(f_positive, tf.floor(rand), tf.math.ceil(rand))

                pow_rand = tf.pow(2.0, rand)
                out_jit = tf.multiply(self.out, pow_rand)
                self.out = tf.where(self.f_fire, out_jit, self.out)

        if self.conf.f_isi:
            self.cal_isi(self.f_fire, t)

    #
    def fire_temporal(self, t):

        time = self.relative_time_fire(t)

        #
        # encoding
        # dynamic threshold (vth)
        #
        self.set_vth_temporal_kernel(time)

        if self.conf.f_refractory:

            # TODO: check - case 1 or 2
            # case 1
            # self.f_fire = (self.vmem >= self.vth) & \
            #                tf.equal(self.refractory,tf.zeros(self.refractory.shape))
            self.f_fire = (self.vmem >= self.vth) & \
                          tf.equal(self.refractory, tf.constant(0.0, tf.float32, self.refractory.shape))

            # case 2
            ##self.f_fire = (self.vmem >= self.vth) & \
            ##                tf.equal(self.refractory,tf.zeros(self.refractory.shape))
            # f_fire = tf.greater_equal(self.vmem,self.vth)
            ##f_refractory = tf.greater_equal(tf.constant(t,tf.float32),self.refractory)
            ##f_refractory = tf.greater_equal(tf.constant(t,tf.float32),self.refractory)
            # f_refractory = tf.greater(tf.constant(t,tf.float32),self.refractory)
            ##self.f_fire = (self.vmem >= self.vth) & \
            ##                  tf.equal(self.refractory,tf.constant(0.0,tf.float32,self.refractory.shape))
            # self.f_fire = tf.logical_and(f_fire,f_refractory)

        else:
            self.f_fire = (self.vmem >= self.vth) & (self.vth >= 10 ** (-5))

        #
        # reset
        #

        # reset by zero
        self.out = tf.where(self.f_fire, tf.ones(self.out.shape), tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire, tf.zeros(self.out.shape), self.vmem)

        # reset by subtraction
        # self.out = tf.where(self.f_fire,self.vth,tf.zeros(self.out.shape))
        # self.vmem = tf.subtract(self.vmem,self.out)

        # addr=0,10,10,0
        # print("fire: glb {}: loc {} - vth {:0.3f}, kernel {:.03f}, out {:0.3f}".format(t, time,self.vmem[addr],self.vth[addr],self.out[addr]))

        if self.conf.f_refractory:
            # self.cal_refractory_temporal(self.f_fire)
            self.cal_refractory_temporal(t)

    #
    def fire_non_lin(self, t):

        #
        self.f_fire = self.vmem >= self.vth
        self.out = tf.where(self.f_fire, self.fires, self.zeros)

        #
        # rand = tf.random_uniform(shape=self.vmem.shape,minval=0.90*self.vth,maxval=self.vth)
        # self.f_fire = tf.logical_or(self.vmem >= self.vth,self.vmem >= rand)

        # self.out = tf.where(self.f_fire,self.fires,self.zeros)

        # reset by subtract
        # self.vmem = tf.subtract(self.vmem,self.out)

        # reset to zero
        # self.out = tf.where(self.f_fire,tf.ones(self.out.shape),tf.zeros(self.out.shape))
        self.vmem = tf.where(self.f_fire, tf.zeros(self.out.shape), self.vmem)

        if self.conf.f_isi:
            self.cal_isi(self.f_fire, t)

    def fire_type_out(self, t):
        f_fire = self.vmem >= self.vth

        self.vmem = tf.where(f_fire, self.vmem - self.vth, self.vmem)

        self.out = tf.where(f_fire, tf.constant(1.0, tf.float32, self.out.shape), tf.zeros(self.vmem.shape))

        # self.isi = tf.where(f_fire,tf.constant(t,tf.float32,self.isi.shape)-self.last_spike_time,self.isi)
        # self.last_spike_time = tf.where(f_fire,tf.constant(t,tf.float32,self.last_spike_time.shape),self.last_spike_time)

    ############################################################
    ##
    ############################################################

    def cal_refractory(self, f_fire):
        f_refractory_update = np.logical_and(np.not_equal(self.vth - self.vth_init, 0.0), np.logical_not(f_fire))
        refractory_update = 2.0 * np.log2(self.vth / self.vth_init)

        self.refractory = tf.maximum(self.refractory - 1, tf.constant(0.0, tf.float32, self.refractory.shape))

        self.refractory = tf.where(f_refractory_update, refractory_update, self.refractory)

        # print(tf.reduce_max(self.vth))
        # print(self.vth_init)
        # print(np.not_equal(self.vth,self.vth_init))
        # print(np.logical_not(f_fire))
        # print(f_refractory_update)
        # self.refractory = tf.where(f_fire,tf.constant(0.0,tf.float32,self.refractory.shape),self.refractory)
        # print(self.refractory)

        # print(tf.reduce_max(np.log2(self.vth/self.vth_init)))

    # TODO: refractory
    def cal_refractory_temporal_original(self, f_fire):
        # self.refractory = tf.where(f_fire,tf.constant(self.conf.time_step,tf.float32,self.refractory.shape),self.refractory)
        self.refractory = tf.where(f_fire, tf.constant(10000.0, tf.float32, self.refractory.shape), self.refractory)

    def cal_refractory_temporal(self, t):
        if self.conf.noise_robust_en:

            inf_t = 10000.0

            t_b = self.conf.noise_robust_spike_num
            # t_b = 2

            f_init_refractory = tf.equal(self.t_set_refractory,
                                         tf.constant(-1.0, dtype=tf.float32, shape=self.t_set_refractory.shape))

            f_first_spike = tf.logical_and(f_init_refractory, self.f_fire)

            t = tf.constant(t, dtype=tf.float32, shape=self.refractory.shape)
            t_b = tf.constant(t_b, dtype=tf.float32, shape=self.refractory.shape)
            inf_t = tf.constant(inf_t, dtype=tf.float32, shape=self.refractory.shape)

            self.t_set_refractory = tf.where(f_first_spike, tf.add(t, t_b), self.t_set_refractory)
            # self.t_set_refractory = tf.where(f_first_spike,t,self.t_set_refractory)

            #
            f_add_vmem = self.f_fire
            # add_amount = tf.divide(self.vth,2.0)
            add_amount = self.vth
            self.vmem = tf.where(f_add_vmem, tf.add(self.vmem, add_amount), self.vmem)

            #
            f_set_inf_refractory = tf.equal(t, self.t_set_refractory)
            self.refractory = tf.where(f_set_inf_refractory, inf_t, self.refractory)

            # print(self.t_set_refractory)

            #
            # self.refractory = tf.where(self.f_fire,tf.constant(10000.0,tf.float32,self.refractory.shape),self.refractory)
            # f_set_refractory = tf.logical_and(self.f_fire,tf.equal(self.refractory,tf.zeros(shape=self.refractory.shape)))
            # self.t_set_refractory = tf.where(f_set_refractory,t,self.t_set_refractory)

            # self.refractory = tf.where(self.f_fire,inf_t,self.refractory)

            # print(type(t.numpy()[0,0,0,0]))
            # print(type(self.t_set_refractory.numpy()[0,0,0,0]))
            # print(t.shape)
            # print(self.t_set_refractory.shape)
            # f_refractory = tf.equal(t,self.t_set_refractory)
            # print(self.t_set_refractory[0])
            # self.refractory = tf.where(f_refractory,inf_t,self.refractory)

            # print(t_int)
            # print(self.depth)
            # print(self.t_set_refractory.numpy()[0,0,0,0:10])
            # print(f_init_refractory.numpy()[0,0,0,0:10])
            # print(self.refractory.numpy()[0,0,0,0:10])
        else:
            self.refractory = tf.where(self.f_fire, tf.constant(10000.0, tf.float32, self.refractory.shape),
                                       self.refractory)

    #
    def count_spike(self, t):
        {
            'TEMPORAL': self.count_spike_temporal
        }.get(self.neural_coding, self.count_spike_default)(t)

    def count_spike_default(self, t):
        # self.spike_count_int = tf.where(self.f_fire,self.spike_count_int+1.0,self.spike_count_int)
        # self.spike_count = tf.add(self.spike_count, self.out)

        self.spike_count_int.assign(tf.where(self.f_fire, self.spike_count_int + 1.0, self.spike_count_int))
        self.spike_count.assign(tf.add(self.spike_count, self.out))

        #print('out')
        #print(self.out)

        ## here
        #print(self.spike_count)
        #print(self.out)

    def count_spike_temporal(self, t):
        self.spike_count_int = tf.add(self.spike_count_int, self.out)
        self.spike_count = tf.where(self.f_fire, tf.add(self.spike_count, self.vth), self.spike_count_int)

        if self.conf.f_record_first_spike_time:
            self.record_first_spike_time(t)

    def record_first_spike_time(self, t):
        # spike_time = self.relative_time_fire(t)
        spike_time = t

        #print(spike_time)

        self.first_spike_time = tf.where(
            #tf.math.logical_and(self.f_fire,tf.equal(self.first_spike_time,self.init_first_spike_time)),
            tf.math.logical_and(self.f_fire,tf.math.is_nan(self.first_spike_time)),
            #tf.constant(spike_time, dtype=tf.float32, shape=self.first_spike_time.shape),
            tf.constant(spike_time, dtype=tf.int32, shape=self.first_spike_time.shape),
            self.first_spike_time)

        #print(tf.reduce_mean(self.first_spike_time))

    ############################################################
    ## run fwd pass
    ############################################################

    def run_type_in(self, inputs, t, training):
        # print('run_type_in')
        self.input_spike_gen(inputs, t)
        self.count_spike(t)
        return self.out

    #
    def run_type_if(self, inputs, t, training):
        #self.leak()
        self.integration(inputs, t)
        self.fire(t)
        self.count_spike(t)

        if glb.model_compiled and self.conf.vth_toggle:
            #print(self.name)
            # toggle_const = tf.where(tf.greater_equal(self.vth,self.fires),1/3,3)
            # toggle_const = tf.where(tf.greater_equal(self.vth,self.fires),0.3/1.7,1.7/0.3)
            # toggle = self.conf.vth_toggle_init
            # toggle_1 = 2-self.conf.vth_toggle_init
            # toggle_const = tf.where(tf.greater_equal(self.vth,self.fires),toggle/toggle_1,toggle_1/toggle)

            # self.vth = tf.where(self.f_fire, self.vth*toggle_const, self.vth)

            # self.vth.assign(tf.where(self.f_fire, self.vth*toggle_const, self.vth))
            #indices = tf.reshape(indices,[-1])

            #idx = self.spike_count_int.numpy()%len(self.vth_schedule)
            #print(idx)
            #self.vth.assign(self.vth_schedule[idx])
            #print(self.vth_schedule)

            # layer or model wise toggle value set
            # neuron-wise toggle value
            toggle_time=16
            if True and (t<toggle_time):
            #if True:

                #indices = tf.cast(tf.math.floormod(self.spike_count_int, len(self.vth_schedule)), dtype=tf.int32)
                indices = tf.cast(tf.math.floormod(self.spike_count_int, len(self.vth_schedule[-1])),dtype=tf.int32)
                #self.indices = indices

                # adaptive vth_schedule
                if False:
                #if True:
                    n_decay_vth_schedule = tf.reshape(tf.math.floordiv(self.spike_count_int, len(self.vth_schedule)),shape=-1)
                    #n_decay_vth_schedule = tf.stack([n_decay_vth_schedule,n_decay_vth_schedule],axis=-1)
                    decay_vth_schedule = tf.math.pow(tf.constant(0.5,shape=n_decay_vth_schedule.shape),n_decay_vth_schedule)
                    decay_vth_schedule = (tf.ones(decay_vth_schedule.shape)-self.vth_toggle_init)*decay_vth_schedule
                    vth_schedule_a = tf.ones(decay_vth_schedule.shape)-decay_vth_schedule
                    vth_schedule_b = vth_schedule_a/(2*vth_schedule_a-1)        # harmonic mean
                    update_vth_schedule = tf.stack([vth_schedule_a,vth_schedule_b],axis=-1)
                    #self.vth_schedule = tf.ones(self.vth_schedule.shape)-decay_vth_schedule
                    self.vth_schedule = update_vth_schedule
                    #self.vth_schedule = self.vth_schedule*decay_vth_schedule


                #
                #self.vth_schedule = tf.where(indices==0, self.vth_schedule/2, self.vth_schedule)

                indices = tf.reshape(indices,shape=-1)
                ind = tf.range(indices.shape)
                indices = tf.stack([ind,indices],axis=-1)

                vth = tf.gather_nd(self.vth_schedule,indices)
                self.vth.assign(tf.reshape(vth,shape=self.vth.shape))
                #self.vth.assign(tf.gather(self.vth_schedule, indices))


            elif t==toggle_time:
                #self.vth_init = tf.constant(self.conf.n_init_vth,shape=self.dim,dtype=tf.float32, name='vth_init')
                self.reset_vth()

        #self.out = tf.nn.relu(inputs)

        #self.out = inputs

    #
    def run_type_lif(self, inputs, t, training):
        # print('run_type_lif')
        self.leak()
        self.integration(inputs, t)
        self.fire(t)
        self.count_spike(t)

        def grad(upstream):
            # TODO: parameterize
            a=0.5
            if True:
            #if False:
                cond_1=tf.math.less_equal(tf.math.abs(self.vmem_pre-self.vth),a)
                #cond_1 = tf.math.logical_and(cond_1,tf.math.logical_not(self.f_fire))
                #cond_2 = self.f_fire
                #cond = tf.math.logical_or(cond_1,cond_2)
                cond = cond_1
                do_du = tf.where(cond,tf.ones(cond.shape),tf.zeros(cond.shape))

            grad_ret = upstream*do_du

            return grad_ret, tf.stop_gradient(t), tf.stop_gradient(training)

        #return self.out, grad
        return self.out
        #print('here')
        #print(t)
        #return self.spike_count/tf.cast(t,tf.float32)
        #return self.spike_count


    def run_type_out(self, inputs, t, training):
        # print("output layer")
        # self.integration(inputs,t)
        ##self.fire_type_out(t)

        if self.conf.snn_output_type in ('SPIKE', 'FIRST_SPIKE_TIME'):
            # in current implementation, output layer acts as IF neuron.
            # If the other types of neuron is needed for the output layer,
            # the declarations of neuron layers in other files should be modified.
            #assert False, 'only support IF?'
            if self.conf.n_type=='IF':
                self.run_type_if(inputs, t, training)
            elif self.conf.n_type=='LIF':
                self.run_type_lif(inputs, t, training)
            else:
                assert False

            self.out = self.spike_count/tf.cast(t,tf.float32)

        else:
            self.integration(inputs, t)
            if self.conf.n_type=='LIF':
                self.leak()
            self.out = self.vmem
            #self.out = tf.divide(self.vmem,self.conf.time_step)
            #out = tf.divide(self.vmem,self.vth)
            #self.out = tf.divide(self.vmem,self.conf.time_step*self.vth)

#        #if self.conf.train:
#        if training:
#            #print('logits')
#            #print(self.out)
#            self.out = tf.keras.activations.softmax(self.out)
#            #self.out = tf.keras.activations.softmax(out)
#            #sm = tf.keras.layers.Softmax()
#            #self.out = sm(self.out)
#        #else:
#            #self.out = out

#

        #self.out = tf.cond(training,lambda:tf.keras.activations.softmax(self.out),lambda:tf.identity(self.out))
        #self.out = tf.keras.activations.softmax(self.out)
        return self.out


    ############################################################
    ##
    ############################################################

    def set_vth(self, vth):
        # self.vth = self.vth.assign(vth)
        self.vth.assign(vth)

    def set_vth_init(self, vth_init):
        self.vth_init = tf.constant(vth_init,shape=self.dim,dtype=tf.float32, name='vth_init')

    def get_spike_count(self):
        #spike_count = tf.reshape(self.spike_count, self.dim)
        #print(self.spike_count)
        return self.spike_count

    def get_spike_count_int(self):
        # spike_count_int = tf.reshape(self.spike_count_int,self.dim)
        return self.spike_count_int

    def get_spike_rate(self):
        # return self.get_spike_count_int()/self.conf.time_step
        return self.get_spike_count_int() / self.conf.time_step

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

    #
    def set_time_const_integ(self, time_const_integ):
        self.time_const_integ = time_const_integ

    def set_time_const_fire(self, time_const_fire):
        self.time_const_fire = time_const_fire

    def set_time_delay_integ(self, time_delay_integ):
        self.time_delay_integ = time_delay_integ

    def set_time_delay_fire(self, time_delay_fire):
        self.time_delay_fire = time_delay_fire

    # def set_time_const_integ(self, time_const_integ):
    #    self.time_const_integ = time_const_integ

    # def set_time_delay_integ(self, time_delay_integ):
    #    self.time_delay_integ = time_delay_integ

    def set_time_integ(self, time_start_integ):
        self.time_start_integ = time_start_integ
        self.time_end_integ = self.time_start_integ + self.conf.time_fire_duration

    def set_time_fire(self, time_start_fire):
        self.time_start_fire = time_start_fire
        self.time_end_fire = self.time_start_fire + self.conf.time_fire_duration

    ############################################################
    ## training time constant (tau) for TTFS coding
    ## gradient-based optimization (DAC-20)
    ############################################################
    def train_time_const_fire(self, dnn_act):
        # print("snn_lib: train_time_const")
        # print(dnn_act)
        # self.time_const_integ = tf.zeros([])
        # self.time_const_fire = tf.multiply(self.time_const_fire,0.1)

        # delta - -1/(2tau^2)(x-x_hat)(x_hat)

        # spike_time = self.first_spike_time-self.depth*self.conf.time_fire_start*self.time_const_fire-self.time_delay_fire

        # if self.conf.f_tc_based:
        #    spike_time = self.first_spike_time-self.depth*self.conf.n_tau_fire_start*self.time_const_fire-self.time_delay_fire
        # else:
        #    spike_time = self.first_spike_time-self.depth*self.conf.time_fire_start-self.time_delay_fire

        spike_time = self.relative_time_fire(self.first_spike_time)
        # spike_time = self.first_spike_time
        spike_time_sub_delay = spike_time - self.time_delay_fire

        x = dnn_act

        # x_hat = tf.where(
        #            tf.equal(self.first_spike_time,tf.constant(self.init_first_spike_time,shape=self.first_spike_time.shape,dtype=tf.float32)), \
        #            tf.zeros(self.first_spike_time.shape), \
        #            tf.exp(-(spike_time_sub_delay/self.time_const_fire)))

        x_hat = tf.where(
            self.flag_fire(), \
            tf.zeros(self.first_spike_time.shape), \
            tf.exp(-(spike_time_sub_delay / self.time_const_fire)))

        # x_hat = tf.exp(-self.first_spike_time/self.time_const_fire)

        # loss = tf.reduce_sum(tf.square(x-x_hat))

        # print(x[0])
        # print(x_hat[0])
        # print(tf.reduce_min(x_hat))
        # print(tf.reduce_max(x_hat))
        # print(tf.reduce_min(spike_time_sub_delay))
        # print(tf.reduce_max(spike_time_sub_delay))

        loss_prec = tf.reduce_mean(tf.square(x - x_hat))
        loss_prec = loss_prec / 2.0

        # l2
        delta1 = tf.subtract(x, x_hat)
        delta1 = tf.multiply(delta1, x_hat)
        # delta1 = tf.multiply(delta1, tf.subtract(self.first_spike_time,self.time_delay_fire))
        delta1 = tf.multiply(delta1, spike_time_sub_delay)

        if tf.equal(tf.size(tf.boolean_mask(delta1, delta1 > 0)), 0):
            delta1 = tf.zeros([])
        else:
            delta1 = tf.reduce_mean(tf.boolean_mask(delta1, delta1 > 0))

        dim = tf.size(tf.shape(x))
        if tf.equal(dim, tf.constant(4)):
            reduce_axis = [1, 2, 3]
        elif tf.equal(dim, tf.constant(3)):
            reduce_axis = [1, 2]
        elif tf.equal(dim, tf.constant(2)):
            reduce_axis = [1]

        if self.conf.f_train_tk_outlier:
            # x_min = tf.tfp.stats.percentile(tf.boolean_mask(x,x>0),0.01)
            # x_min = tf.constant(np.percentile(tf.boolean_mask(x,x>0).numpy(),1),dtype=tf.float32,shape=[])

            x_pos = tf.where(x > tf.zeros(x.shape), x, tf.zeros(x.shape))
            x_min = tf.constant(np.percentile(x_pos.numpy(), 2, axis=reduce_axis), dtype=tf.float32,
                                shape=x_pos.shape[0])

            # print("min: {:e}, min_0.01: {:e}".format(tf.reduce_min(tf.boolean_mask(x,x>0)),x_min))
        else:
            # ~x_min = tf.reduce_min(tf.boolean_mask(x,x>0))
            x_pos = tf.where(x > tf.zeros(x.shape), x, tf.zeros(x.shape))
            x_min = tf.reduce_min(x_pos, axis=reduce_axis)

        if self.conf.f_tc_based:
            fire_duration = self.conf.n_tau_fire_duration * self.time_const_fire
        else:
            fire_duration = self.conf.time_fire_duration

        # x_hat_min = tf.exp(-(self.conf.time_fire_duration/self.time_const_fire))
        x_hat_min = tf.exp(-(fire_duration - self.time_delay_fire) / self.time_const_fire)

        loss_min = tf.reduce_mean(tf.square(x_min - x_hat_min))
        loss_min = loss_min / 2.0

        x_min = tf.reduce_mean(x_min)

        delta2 = tf.subtract(x_min, x_hat_min)
        delta2 = tf.multiply(delta2, x_hat_min)

        if self.conf.f_tc_based:
            delta2 = tf.multiply(delta2,
                                 tf.subtract(self.conf.n_tau_time_window * self.time_const_fire, self.time_delay_fire))
        else:
            delta2 = tf.multiply(delta2, tf.subtract(self.conf.time_window, self.time_delay_fire))

        # delta2 = tf.reduce_mean(delta2)

        #
        # idx=0,0,0,0
        # print("x: {:e}, x_hat: {:e}".format(x[idx],x_hat[idx]))
        # print("x_min: {:e}, x_hat_min: {:e}".format(x_min,x_hat_min))

        # l1

        #
        delta1 = tf.divide(delta1, tf.square(self.time_const_fire))
        delta2 = tf.divide(delta2, tf.square(self.time_const_fire))

        # rho1 = 10.0
        # rho2 = 100.0

        rho1 = 1.0
        rho2 = 1.0

        #
        delta = tf.add(tf.multiply(delta1, rho1), tf.multiply(delta2, rho2))

        # print("name: {:s}, del: {:e}, del1: {:e}, del2: {:e}".format(self.name,delta,delta1,delta2))

        # self.time_const_fire = tf.subtract(self.time_const_fire, delta)
        self.time_const_fire = tf.add(self.time_const_fire, delta)

        #
        # idx=0,10,10,0
        # print('x: {:e}, vmem: {:e}, x_hat: {:e}, delta: {:e}'.format(x[idx],self.vmem[idx],x_hat[idx],delta))

        print("name: {:s}, loss_prec: {:g}, loss_min: {:g}, tc: {:f}".format(self.name, loss_prec, loss_min,
                                                                             self.time_const_fire))

        self.loss_prec = loss_prec
        self.loss_min = loss_min

        #
        # print("name: {:s}, tc: {:f}".format(self.name,self.time_const_fire))

        # print("\n")

    ############################################################
    ## training time delay (td) for TTFS coding
    ## gradient-based optimization (DAC-20)
    ############################################################
    def train_time_delay_fire(self, dnn_act):

        #        if self.conf.f_train_tk_outlier:
        #            t_ref = self.depth*self.conf.time_fire_start
        #            t_min = np.percentile(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0).numpy(),0.01)
        #            t_min = t_min-t_ref
        #        else:
        #            t_ref = self.depth*self.conf.time_fire_start
        #            t_min = tf.reduce_min(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0))
        #            t_min = t_min-t_ref

        # t_ref = self.depth*self.conf.time_fire_start*self.time_const_fire

        if self.conf.f_tc_based:
            t_ref = self.depth * self.conf.n_tau_fire_start * self.time_const_fire
        else:
            t_ref = self.depth * self.conf.time_fire_start

        dim = tf.size(tf.shape(self.first_spike_time))
        if tf.equal(dim, tf.constant(4)):
            reduce_axis = [1, 2, 3]
        elif tf.equal(dim, tf.constant(3)):
            reduce_axis = [1, 2]
        elif tf.equal(dim, tf.constant(2)):
            reduce_axis = [1]

        # print(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0,keepdims=True).shape)

        # t_min = tf.reduce_min(tf.boolean_mask(self.first_spike_time,self.first_spike_time>0))
        t_min = tf.where(tf.equal(self.first_spike_time, self.init_first_spike_time),
                         tf.constant(99999.9, shape=self.first_spike_time.shape), self.first_spike_time)
        t_min = tf.reduce_min(t_min, axis=reduce_axis)
        # t_min = t_min-t_ref
        t_min = self.relative_time_fire(t_min)

        x_max = tf.exp(-(t_min - self.time_delay_fire) / self.time_const_fire)

        x_max_hat = tf.exp(self.time_delay_fire / self.time_const_fire)

        loss_max = tf.reduce_mean(tf.square(x_max - x_max_hat))
        loss_max = loss_max / 2.0

        delta = tf.subtract(x_max, x_max_hat)
        delta = tf.multiply(delta, x_max_hat)
        delta = tf.divide(delta, self.time_const_fire)
        delta = tf.reduce_mean(delta)

        rho = 1.0

        delta = tf.multiply(delta, rho)

        # print(self.first_spike_time)
        # print(t_min)
        ##print(x_max)
        # print(x_max_hat)

        # print("t_min: {:f}".format(t_min))
        # print("x_max: {:f}".format(x_max))
        # print("x_max_hat: {:f}".format(x_max_hat))
        # print("delta: {:f}".format(delta))

        # self.time_delay_fire = tf.subtract(self.time_delay_fire,delta)
        self.time_delay_fire = tf.add(self.time_delay_fire, delta)

        # print("name: {:s}, del: {:e}, td: {:e}".format(self.name,delta,self.time_delay_fire))
        print("name: {:s}, loss_max: {:e}, td: {:f}".format(self.name, loss_max, self.time_delay_fire))

        self.loss_max = loss_max

    ############################################################
    ## This function is needed for fire phase in the temporal coding (TTFS)
    ## time converter: absolute time (global time) -> relative time in each time window (local time)
    ## t: absolute time (global time)
    ## time: relative time (local time)
    ############################################################
    def relative_time_fire(self, t):
        assert False, 'add code when t is negative (no spike)'
        if self.conf.f_tc_based:
            time = t - self.depth * self.conf.n_tau_fire_start * self.time_const_fire
        else:
            time = t - self.depth * self.conf.time_fire_start
        return time

    ############################################################
    ## This function is needed for integration phase in the temporal coding (TTFS)
    ## time converter: absolute time (global time) -> relative time in each time window (local time)
    ## t: absolute time (global time)
    ## time: relative time (local time)
    ############################################################
    def relative_time_integ(self, t):
        if self.conf.f_tc_based:
            time = t - (self.depth - 1) * self.conf.n_tau_fire_start * self.time_const_integ
        else:
            time = t - (self.depth - 1) * self.conf.time_fire_start
        return time

    #
    def flag_fire(self):
        ret = tf.not_equal(self.spike_count, tf.constant(0.0, tf.float32, self.spike_count.shape))
        return ret

    ###########################################################################
    ## SNN training w/ TTFS coding
    ###########################################################################


###############################################################################
## Temporal kernel for surrogate model training
## enc(t)=ta*exp(-(t-td)/tc)
###############################################################################
class Temporal_kernel(tf.keras.layers.Layer):
    def __init__(self, dim_in, dim_out, conf):
        super(Temporal_kernel, self).__init__()

        #
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dim_in_one_batch = [1, ] + dim_in[1:]
        self.dim_out_one_batch = [1, ] + dim_out[1:]

        #
        self.init_tc = conf.tc
        self.init_td = self.init_tc * np.log(conf.td)
        self.init_tw = conf.time_window

        self.epoch_start_t_int = conf.epoch_start_train_t_int
        self.epoch_start_clip_tw = conf.epoch_start_train_clip_tw
        self.epoch_start_train_tk = conf.epoch_start_train_tk
        # start epoch training with floor function - quantization
        # before this epoch, training with round founction
        self.epoch_start_train_floor = conf.epoch_start_train_floor

        #
        # self.enc_st_n_tw = conf.enc_st_n_tw
        # encoding maximum spike time
        self.ems_mode = conf.ems_loss_enc_spike
        self.ems_nt_mult_tw = conf.enc_st_n_tw * conf.time_window

        #
        self.f_td_training = conf.f_td_training

        # encoding decoding para couple
        self.f_enc_dec_couple = True
        # self.f_enc_dec_couple = False

        # double tc
        # self.f_double_tc = True
        self.f_double_tc = False

    def build(self, _):

        # TODO: parameterize
        # which one ?
        # a para per layer
        # neuron-wise para
        self.tc = self.add_variable("tc", shape=self.dim_in_one_batch, dtype=tf.float32,
                                    initializer=tf.constant_initializer(self.init_tc), trainable=True)
        # self.td = self.add_variable("td",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_td),trainable=True)
        self.td = self.add_variable("td", shape=self.dim_in_one_batch, dtype=tf.float32,
                                    initializer=tf.constant_initializer(self.init_td), trainable=self.f_td_training)
        self.tw = self.add_variable("tw", shape=self.dim_in_one_batch, dtype=tf.float32,
                                    initializer=tf.constant_initializer(self.init_tw), trainable=False)

        if self.f_double_tc:
            self.tc_1 = self.add_variable("tc_1", shape=self.dim_in_one_batch, dtype=tf.float32,
                                          initializer=tf.constant_initializer(10.0), trainable=True)
            self.td_1 = self.add_variable("td_1", shape=self.dim_in_one_batch, dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0), trainable=True)

        #
        # decoding para
        # self.tc_dec = self.add_variable("tc_dec",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tc),trainable=True)
        # self.td_dec = self.add_variable("td_dec",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_td),trainable=True)
        ##self.tw_dec = self.add_variable("tw_dec",shape=self.dim_in_one_batch,dtype=tf.float32,initializer=tf.constant_initializer(self.init_tw),trainable=False)

        #
        # decoding para

        if not self.f_enc_dec_couple:
            self.tc_dec = self.add_variable("tc_dec", shape=self.dim_out_one_batch, dtype=tf.float32,
                                            initializer=tf.constant_initializer(self.init_tc), trainable=True)
            self.td_dec = self.add_variable("td_dec", shape=self.dim_out_one_batch, dtype=tf.float32,
                                            initializer=tf.constant_initializer(self.init_td), trainable=True)
            self.tw_dec = self.add_variable("tw_dec", shape=self.dim_out_one_batch, dtype=tf.float32,
                                            initializer=tf.constant_initializer(self.init_tw), trainable=False)

        # input - encoding target
        self.in_enc = self.add_variable("in_enc", shape=self.dim_out, dtype=tf.float32,
                                        initializer=tf.zeros_initializer(), trainable=False)
        # output of encoding - spike time
        self.out_enc = self.add_variable("out_enc", shape=self.dim_out, dtype=tf.float32,
                                         initializer=tf.zeros_initializer(), trainable=False)
        # output of decoding
        self.out_dec = self.add_variable("out_dec", shape=self.dim_in, dtype=tf.float32,
                                         initializer=tf.zeros_initializer(), trainable=False)

    def call_tmp_question(self, input, mode, epoch, training=None):
        mode_sel = {
            'enc': self.call_encoding,
            'dec': self.call_decoding
        }
        ret = mode_sel[mode](input, epoch, training)

        return ret

    def call_encoding(self, input, epoch, training):

        #
        self.in_enc = input

        #
        t_float = self.call_encoding_kernel(input)

        #
        infer_mode = (training == False) and (epoch < 0)
        #
        # if False:
        # if ((training==False) and (epoch==-1)) or ((training == True) and (epoch > self.epoch_start_t_int)):
        if ((training == True) and (tf.math.greater(epoch, self.epoch_start_t_int))) or (training == False):
            # TODO: parameterize
            # if epoch > self.epoch_start_t_int+100:
            # if epoch > self.epoch_start_t_int:
            #    t = tf.ceil(t_float)
            # else:
            #    t = tf.quantization.fake_quant_with_min_max_vars(t_float,0,tf.pow(2.0,16.0)-1,16)
            #   #` t=tf.round(t_float)

            # if (epoch < self.epoch_start_train_floor) and not infer_mode:
            if ((training == True) and (tf.math.greater(epoch, self.epoch_start_train_floor))) or (training == False):
                # t = tf.quantization.fake_quant_with_min_max_vars(t_float,0,tf.pow(2.0,16.0)-1,16)
                # t = tf.math.add(tf.math.floor(t_float),1)
                t = tf.math.ceil(t_float)
                # t = tf.math.ceil(t)
            else:
                t = tf.quantization.fake_quant_with_min_max_vars(t_float, 0, tf.pow(2.0, 16.0) - 1, 16)

            # tmp = tf.where(tf.equal(t,0),100,t)
            # print(tf.reduce_min(tmp))

        else:
            t = t_float

        #        #
        #        if (training == False) or ((training==True) and (epoch > self.epoch_start_clip_tw)):
        #        #if False:
        #        #if True:
        #            #print(t)
        #            t=tf.math.minimum(t, self.tw)
        #            #print(self.tw)

        # print('min: {:}, max:{:}'.format(tf.reduce_min(t),tf.reduce_max(t)))
        # print(t)

        #
        self.out_enc = t

        # print(tf.reduce_mean(self.tc))
        # print(tf.reduce_mean(self.td))
        # print(tf.reduce_mean(self.ta))

        return t

    def call_encoding_kernel(self, input):

        if self.ems_mode == 'f':
            eps = 1.0E-30
        elif self.ems_mode == 'n':
            # eps = tf.math.exp(-float(self.ems))
            eps = tf.math.exp(tf.math.divide(tf.math.subtract(self.td, self.ems_nt_mult_tw), self.tc))
        else:
            assert False, 'not supported encoding maximum spike mode - {}'.format(self.ems_mode)

        # x = tf.nn.relu(input)
        # x = tf.divide(x,self.ta)
        x = tf.nn.relu(input)
        x = tf.add(x, eps)
        x = tf.math.log(x)

        if self.f_double_tc:
            # t = tf.subtract(self.td, tf.multiply(x,self.tc))
            A = tf.math.log(tf.add(tf.exp(tf.divide(self.td, self.tc)), tf.exp(tf.divide(self.td_1, self.tc_1))))
            x = tf.subtract(A, x)
            t = tf.multiply(tf.divide(tf.add(self.tc, self.tc_1), 2), x)
        else:
            t = tf.subtract(self.td, tf.multiply(x, self.tc))

        t = tf.nn.relu(t)

        # print(t)

        # print(t)
        # print(self.td)

        return t

    def call_decoding(self, t, epoch, training):

        # x = tf.multiply(self.ta,tf.exp(tf.divide(tf.subtract(self.td,t),self.tc)))

        if self.f_enc_dec_couple:
            tw_target = self.tw
            td = self.td
            tc = self.tc

            # if epoch > 500:
            #    tw_target = 1.5*self.tw - self.tw/1000*epoch

        else:
            # tw_target = self.tw_dec
            # td = self.td_dec
            # tc = self.tc_dec

            tw_target = self.tw
            td = self.td
            tc = self.tc
        #
        if self.f_double_tc:
            x = tf.add(tf.exp(tf.divide(tf.subtract(td, t), tc)),
                       tf.exp(tf.divide(tf.subtract(self.td_1, t), self.tc_1)))
        else:
            x = tf.exp(tf.divide(tf.subtract(td, t), tc))

        # if False:
        # if (training == False) or ((training==True) and (epoch > self.epoch_start_clip_tw)):
        # if (training==True)and(epoch > self.epoch_start_clip_tw) or (training==False)and(epoch<0):
        if ((training == True) and (tf.math.greater(epoch, self.epoch_start_clip_tw))) or (training == False):
            # if epoch > 300:
            #    tw_target = self.tw/2
            # else:
            #    tw_target = self.tw

            #
            tk_min = tf.exp(tf.divide(tf.subtract(td, tw_target), tc))

            # print('min: {:}, max:{:}'.format(tf.reduce_min(x),tf.reduce_max(x)))

            # print('')
            # print(tw_target)
            # print(tk_min)

            x_clipped = tf.where(x >= tf.broadcast_to(tk_min, shape=x.shape), \
                                 x, tf.constant(0.0, shape=x.shape, dtype=tf.float32))

            # x_clipped = x-tk_min
            # x_clipped = tf.nn.relu(x_clipped)
            # x_clipped = x_clipped+tk_min

            x = x_clipped

        self.out_dec = x

        # print('min: {:}, max:{:}'.format(tf.reduce_min(x),tf.reduce_max(x)))

        return x

    #
    def set_init_td_by_target_range(self, act_target_range):

        td = tf.multiply(self.tc, tf.math.log(act_target_range))
        self.td.assign(tf.constant(td, dtype=tf.float32, shape=self.td.shape))
