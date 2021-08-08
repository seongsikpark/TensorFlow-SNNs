
import tensorflow_probability as tfp
tfd = tfp.distributions

# TODO: modify and verification
###########################################################################
## T2FSNN
###########################################################################
def T2FSNN_load_time_const(self):
    file_name = self.conf.tk_file_name+'_itr-{:d}'.format(self.conf.time_const_num_trained_data)

    if self.conf.f_train_tk_outlier:
        file_name+="_outlier"

    print('load trained time constant: file_name: {:s}'.format(file_name))

    file = open(file_name,'r')
    lines = csv.reader(file)

    # load tk
    for line in lines:
        if not line:
            continue

        print(line)

        type = line[0]
        name = line[1]
        val = float(line[2])

        if (type=='tc') :

            self.list_neuron[name].set_time_const_init_fire(val)

            if not ('in' in name):
                self.list_neuron[name].set_time_const_init_integ(self.list_neuron[name_prev].time_const_init_fire)

            name_prev = name

        elif (type=='td'):

            self.list_neuron[name].set_time_delay_init_fire(val)

            if not ('in' in name):
                self.list_neuron[name].set_time_delay_init_integ(self.list_neuron[name_prev].time_delay_init_fire)

            name_prev = name

        else:
            print("not supported temporal coding type")
            assert(False)


    file.close()


###########################################################################
##
###########################################################################
#
# TODO: input neuron ?
def load_temporal_kernel_para(self):
    if self.conf.verbose:
        print('preprocessing: load_temporal_kernel_para')

    for l_name in self.list_layer_name:

        if l_name != self.list_layer_name[-1]:
            self.list_neuron[l_name].set_time_const_fire(self.list_tk[l_name].tc)
            self.list_neuron[l_name].set_time_delay_fire(self.list_tk[l_name].td)

        if not ('in' in l_name):
            #self.list_neuron[l_name].set_time_const_integ(self.list_tk[l_name_prev].tc_dec)
            #self.list_neuron[l_name].set_time_delay_integ(self.list_tk[l_name_prev].td_dec)
            self.list_neuron[l_name].set_time_const_integ(self.list_tk[l_name_prev].tc)
            self.list_neuron[l_name].set_time_delay_integ(self.list_tk[l_name_prev].td)

        l_name_prev = l_name


###########################################################
## training time constant for temporal coding
###########################################################

# TODO: verification
# training time constant for temporal coding
def train_time_const(self):

    print("models: train_time_const")

    # train_time_const
    name_layer_prev=''
    for name_layer, layer in self.list_neuron.items():
        if not ('fc3' in name_layer):
            dnn_act = self.dnn_act_list[name_layer]
            self.list_neuron[name_layer].train_time_const_fire(dnn_act)

        if not ('in' in name_layer):
            self.list_neuron[name_layer].set_time_const_integ(self.list_neuron[name_layer_prev].time_const_fire)

        name_layer_prev = name_layer


    # train_time_delay
    name_layer_prev=''
    for name_layer, layer in self.list_neuron.items():
        if not ('fc3' in name_layer or 'in' in name_layer):
            dnn_act = self.dnn_act_list[name_layer]
            self.list_neuron[name_layer].train_time_delay_fire(dnn_act)

        if not ('in' in name_layer or 'conv1' in name_layer):
            self.list_neuron[name_layer].set_time_delay_integ(self.list_neuron[name_layer_prev].time_delay_fire)

        name_layer_prev = name_layer


#        if self.conf.f_tc_based:
#            # update integ and fire time
#            name_layer_prev=''
#            for name_layer, layer in self.list_neuron.items():
#                if not ('fc3' in name_layer):
#                    self.list_neuron[name_layer].set_time_fire(self.list_neuron[name_layer].time_const_fire*self.conf.n_tau_fire_start)
#
#                if not ('in' in name_layer):
#                    self.list_neuron[name_layer].set_time_integ(self.list_neuron[name_layer_prev].time_const_integ*self.conf.n_tau_fire_start)
#
#                name_layer_prev = name_layer


# TODO: verification
def get_time_const_train_loss(self):

    loss_prec=0
    loss_min=0
    loss_max=0

    #for name_layer, layer in self.list_neuron.items():
    #    if not ('fc3' in name_layer):
    #        loss_prec += self.list_neuron[name_layer].loss_prec
    #        loss_min += self.list_neuron[name_layer].loss_min
    #        loss_max += self.list_neuron[name_layer].loss_max

    #
    for layer in self.list_layer:
        if layer != self.output_layer:
            loss_prec += layer.act.loss_prec
            loss_min += layer.act.loss_min
            loss_max += layer.act.loss_max

    return [loss_prec, loss_min, loss_max]

# TODO: verification
###########################################################################
## Surrogate DNN model for training SNN w/ TTFS coding
###########################################################################
def surrogate_training_setup(self):
# surrogate DNN model for SNN training w/ TTFS coding
    if self.conf.f_surrogate_training_model:

        self.list_tk=collections.OrderedDict()

        for l_name, l in self.list_layer.items():
            if (not 'bn' in l_name):
                if (not 'fc3' in l_name):
                    self.list_tk[l_name] = lib_snn.layers.Temporal_kernel([], [], self.conf)

        #
        self.enc_st_n_tw = self.conf.enc_st_n_tw

        self.enc_st_target_end = self.conf.time_window*self.enc_st_n_tw

        # TODO: parameterize with other file (e.g., train_snn.py)
        #f_loss_dist = True

        # TODO: function
        self.f_loss_enc_spike_dist = False
        self.f_loss_enc_spike_bn = False
        self.f_loss_enc_spike_bn_only = False   # loss aginst only BN parameters
        self.f_loss_enc_spike_bn_only_new = False   # debug version
        self.f_loss_enc_spike_bn_only_new_2 = False   # squred
        self.f_loss_enc_spike_bn_only_new_lin = False   # linear approx
        self.f_loss_enc_spike_bn_only_new_new = False   # debug version - new new

        if self.conf.d_loss_enc_spike == 'bn':
            self.f_loss_enc_spike_dist = False
            self.f_loss_enc_spike_bn = True
            self.f_loss_enc_spike_bn_only = False
        elif self.conf.d_loss_enc_spike == 'bno':
            self.f_loss_enc_spike_dist = False
            self.f_loss_enc_spike_bn = True
            self.f_loss_enc_spike_bn_only = True
        elif self.conf.d_loss_enc_spike == 'bnon':
            self.f_loss_enc_spike_dist = False
            self.f_loss_enc_spike_bn = True
            self.f_loss_enc_spike_bn_only = True
            self.f_loss_enc_spike_bn_only_new = True
        elif self.conf.d_loss_enc_spike == 'bnon2':
            self.f_loss_enc_spike_dist = False
            self.f_loss_enc_spike_bn = True
            self.f_loss_enc_spike_bn_only = True
            self.f_loss_enc_spike_bn_only_new_2 = True
        elif self.conf.d_loss_enc_spike == 'bnonl':
            self.f_loss_enc_spike_dist = False
            self.f_loss_enc_spike_bn = True
            self.f_loss_enc_spike_bn_only = True
            self.f_loss_enc_spike_bn_only_new_lin = True
        elif self.conf.d_loss_enc_spike == 'bnonn':
            self.f_loss_enc_spike_dist = False
            self.f_loss_enc_spike_bn = True
            self.f_loss_enc_spike_bn_only = True
            self.f_loss_enc_spike_bn_only_new_new = True
        else:
            self.f_loss_enc_spikes_dist = self.conf.f_loss_enc_spike
            self.f_loss_enc_spike_bn = False
            self.f_loss_enc_spike_bn_only = False

        # TODO: function
        if self.f_loss_enc_spike_dist:

            #alpha = 0.1
            #beta = 0.9

            alpha = self.conf.beta_dist_a
            beta = self.conf.beta_dist_b


            if 'b' in self.conf.d_loss_enc_spike:
                self.dist = tfd.Beta(alpha,beta)
            elif 'g' in self.conf.d_loss_enc_spike:
                self.dist = tfd.Gamma(alpha,beta)
            elif 'h' in self.conf.d_loss_enc_spike:
                self.dist = tfd.Horseshoe(alpha)
            else:
                assert False, 'not supported distribution {}'.format(self.conf.d_loss_enc_spike)

            self.dist_beta_sample = collections.OrderedDict()

        #
        self.train_tk_strategy = self.conf.train_tk_strategy.split('-')[0]
        if self.train_tk_strategy != 'N':
            self.train_tk_strategy_coeff = (int)(self.conf.train_tk_strategy.split('-')[1])
            self.train_tk_strategy_coeff_x3 = self.train_tk_strategy_coeff*3

        #
        self.t_train_tk_reg = self.conf.t_train_tk_reg.split('-')[0]
        self.t_train_tk_reg_mode = self.conf.t_train_tk_reg.split('-')[1]

###########################################################################
##
###########################################################################
def dist_beta_sample_func(self):
    for l_name, tk in self.list_tk.items():
        enc_st = tf.reshape(tk.out_enc, [-1])

        samples = self.dist.sample(enc_st.shape)
        #samples = tf.divide(samples,tf.reduce_max(samples))
        samples = tf.multiply(samples,self.enc_st_target_end)
        self.dist_beta_sample[l_name] = tf.histogram_fixed_width(samples, [0,self.enc_st_target_end], nbins=self.enc_st_target_end)



###########################################################################
## Surrogate DNN model for training SNN w/ TTFS coding
###########################################################################
def call_ann_surrogate_training(self,inputs,training,tw,epoch):
    #print(epoch)
    #print(type(inputs))
    #if self.f_1st_iter == False and self.conf.nn_mode=='ANN':
    if self.f_1st_iter == False:
        #if self.f_done_preproc == False:
        #self.f_done_preproc=True
        #self.print_model_conf()
        #self.preproc_ann_norm()
        self.f_skip_bn=self.conf.f_fused_bn
    else:
        self.f_skip_bn=False

    x = tf.reshape(inputs,self._input_shape)

    #a_in = x

    #pr = 0.1
    #pr = 0.6
    #pr = 0.9
    #pr = 1.0
    #target_epoch = 600

    target_epoch = self.conf.bypass_target_epoch
    pr_target_epoch = tf.cast(tf.divide(tf.add(epoch,1),target_epoch),tf.float32)
    pr = tf.multiply(tf.subtract(1.0,self.conf.bypass_pr),pr_target_epoch)


    #if epoch==-1 or epoch > 100:
    #if epoch==-1 or tf.random.uniform(shape=(),minval=0,maxval=1)<pr:
    #pr_layer = pr*(5/5)*pr_target_epoch

    #pr_layer = tf.multiply(pr,pr_target_epoch)
    pr_layer = pr
    #print("epoch: {}, target_epoch: {}".format(epoch,target_epoch))
    #print("pr: {}, pr_target_epoch: {}".format(pr_layer,pr_target_epoch))
    #pr_layer = 2.0


    #if self.f_1st_iter:
    #        if not self.f_load_model_done:
    #        #if training==False or tf.random.uniform(shape=(),minval=0,maxval=1)<pr_layer:
    #            v_in = x
    #            t_in = self.list_tk['in'](v_in,'enc', self.epoch, training)
    #            v_in_dec= self.list_tk['in'](t_in, 'dec', self.epoch, training)
    #            a_in = v_in_dec
    #        else:
    #            a_in = x
    a_in = x

    s_conv1 = self.list_layer['conv1'](a_in)
    if self.f_skip_bn:
        s_conv1_bn = s_conv1
    else:
        s_conv1_bn = self.list_layer['conv1_bn'](s_conv1,training=training)


    #if epoch==-1 or epoch > 100:
    #if epoch==-1 or tf.random.uniform(shape=(),minval=0,maxval=1)<pr:
    #if training==False or ((training) and (tf.random.uniform(shape=(),minval=0,maxval=1)<pr_layer)):
    #if training==False or ((training) and (rand<pr_layer)):
    rand = tf.random.uniform(shape=(),minval=0,maxval=1)
    if training==False or ((training==True) and (tf.math.less(rand,pr_layer))):
        #if True:
        v_conv1 = s_conv1_bn
        t_conv1 = self.list_tk['conv1'](v_conv1,'enc',self.epoch,training)
        v_conv1_dec = self.list_tk['conv1'](t_conv1,'dec',self.epoch,training)
        a_conv1 = v_conv1_dec
    else:
        a_conv1 = tf.nn.relu(s_conv1_bn)

    if training:
        a_conv1 = self.dropout_conv(a_conv1,training=training)
    s_conv1_1 = self.list_layer['conv1_1'](a_conv1)

    #pred = tf.reduce_mean(self.list_layer['conv1_1'].kernel,[0,1])

    if self.f_skip_bn:
        s_conv1_1_bn = s_conv1_1
    else:
        s_conv1_1_bn = self.list_layer['conv1_1_bn'](s_conv1_1,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv1_1 = s_conv1_1_bn
        t_conv1_1 = self.list_tk['conv1_1'](v_conv1_1,'enc',self.epoch,training)
        v_conv1_1_dec = self.list_tk['conv1_1'](t_conv1_1,'dec',self.epoch,training)
        a_conv1_1 = v_conv1_1_dec
    else:
        a_conv1_1 = tf.nn.relu(s_conv1_1_bn)

    p_conv1_1 = self.pool2d(a_conv1_1)
    #if training:
    #    x = self.dropout_conv(x,training=training)

    s_conv2 = self.list_layer['conv2'](p_conv1_1)
    if self.f_skip_bn:
        s_conv2_bn = s_conv2
    else:
        s_conv2_bn = self.list_layer['conv2_bn'](s_conv2,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv2 = s_conv2_bn
        t_conv2 = self.list_tk['conv2'](v_conv2,'enc',self.epoch,training)
        v_conv2_dec = self.list_tk['conv2'](t_conv2,'dec',self.epoch,training)
        a_conv2 = v_conv2_dec
    else:
        a_conv2 = tf.nn.relu(s_conv2_bn)

    if training:
        a_conv2 = self.dropout_conv2(a_conv2,training=training)
    s_conv2_1 = self.list_layer['conv2_1'](a_conv2)
    if self.f_skip_bn:
        s_conv2_1_bn = s_conv2_1
    else:
        s_conv2_1_bn = self.list_layer['conv2_1_bn'](s_conv2_1,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv2_1 = s_conv2_1_bn
        t_conv2_1 = self.list_tk['conv2_1'](v_conv2_1,'enc',self.epoch,training)
        v_conv2_1_dec = self.list_tk['conv2_1'](t_conv2_1,'dec',self.epoch,training)
        a_conv2_1 = v_conv2_1_dec
    else:
        a_conv2_1 = tf.nn.relu(s_conv2_1_bn)

    p_conv2_1 = self.pool2d(a_conv2_1)
    #if training:
    #   x = self.dropout_conv2(x,training=training)

    s_conv3 = self.list_layer['conv3'](p_conv2_1)
    if self.f_skip_bn:
        s_conv3_bn = s_conv3
    else:
        s_conv3_bn = self.list_layer['conv3_bn'](s_conv3,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv3 = s_conv3_bn
        t_conv3 = self.list_tk['conv3'](v_conv3,'enc',self.epoch,training)
        v_conv3_dec = self.list_tk['conv3'](t_conv3,'dec',self.epoch,training)
        a_conv3 = v_conv3_dec
    else:
        a_conv3 = tf.nn.relu(s_conv3_bn)

    if training:
        a_conv3 = self.dropout_conv2(a_conv3,training=training)
    s_conv3_1 = self.list_layer['conv3_1'](a_conv3)
    if self.f_skip_bn:
        s_conv3_1_bn = s_conv3_1
    else:
        s_conv3_1_bn = self.list_layer['conv3_1_bn'](s_conv3_1,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv3_1 = s_conv3_1_bn
        t_conv3_1 = self.list_tk['conv3_1'](v_conv3_1,'enc',self.epoch,training)
        v_conv3_1_dec = self.list_tk['conv3_1'](t_conv3_1,'dec',self.epoch,training)
        a_conv3_1 = v_conv3_1_dec
    else:
        a_conv3_1 = tf.nn.relu(s_conv3_1_bn)

    if training:
        a_conv3_1 = self.dropout_conv2(a_conv3_1,training=training)
    s_conv3_2 = self.list_layer['conv3_2'](a_conv3_1)
    if self.f_skip_bn:
        s_conv3_2_bn = s_conv3_2
    else:
        s_conv3_2_bn = self.list_layer['conv3_2_bn'](s_conv3_2,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv3_2 = s_conv3_2_bn
        t_conv3_2 = self.list_tk['conv3_2'](v_conv3_2,'enc',self.epoch,training)
        v_conv3_2_dec = self.list_tk['conv3_2'](t_conv3_2,'dec',self.epoch,training)
        a_conv3_2 = v_conv3_2_dec
    else:
        a_conv3_2 = tf.nn.relu(s_conv3_2_bn)

    p_conv3_2 = self.pool2d(a_conv3_2)
    #if training:
    #   x = self.dropout_conv2(x,training=training)

    s_conv4 = self.list_layer['conv4'](p_conv3_2)
    if self.f_skip_bn:
        s_conv4_bn = s_conv4
    else:
        s_conv4_bn = self.list_layer['conv4_bn'](s_conv4,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv4 = s_conv4_bn
        t_conv4 = self.list_tk['conv4'](v_conv4,'enc',self.epoch,training)
        v_conv4_dec = self.list_tk['conv4'](t_conv4,'dec',self.epoch,training)
        a_conv4 = v_conv4_dec
    else:
        a_conv4 = tf.nn.relu(s_conv4_bn)

    if training:
        a_conv4 = self.dropout_conv2(a_conv4,training=training)
    s_conv4_1 = self.list_layer['conv4_1'](a_conv4)
    if self.f_skip_bn:
        s_conv4_1_bn = s_conv4_1
    else:
        s_conv4_1_bn = self.list_layer['conv4_1_bn'](s_conv4_1,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv4_1 = s_conv4_1_bn
        t_conv4_1 = self.list_tk['conv4_1'](v_conv4_1,'enc',self.epoch,training)
        v_conv4_1_dec = self.list_tk['conv4_1'](t_conv4_1,'dec',self.epoch,training)
        a_conv4_1 = v_conv4_1_dec
    else:
        a_conv4_1 = tf.nn.relu(s_conv4_1_bn)

    if training:
        a_conv4_1 = self.dropout_conv2(a_conv4_1,training=training)
    s_conv4_2 = self.list_layer['conv4_2'](a_conv4_1)
    if self.f_skip_bn:
        s_conv4_2_bn = s_conv4_2
    else:
        s_conv4_2_bn = self.list_layer['conv4_2_bn'](s_conv4_2,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv4_2 = s_conv4_2_bn
        t_conv4_2 = self.list_tk['conv4_2'](v_conv4_2,'enc',self.epoch,training)
        v_conv4_2_dec = self.list_tk['conv4_2'](t_conv4_2,'dec',self.epoch,training)
        a_conv4_2 = v_conv4_2_dec
    else:
        a_conv4_2 = tf.nn.relu(s_conv4_2_bn)

    p_conv4_2 = self.pool2d(a_conv4_2)
    #if training:
    #   x = self.dropout_conv2(x,training=training)

    s_conv5 = self.list_layer['conv5'](p_conv4_2)
    if self.f_skip_bn:
        s_conv5_bn = s_conv5
    else:
        s_conv5_bn = self.list_layer['conv5_bn'](s_conv5,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv5 = s_conv5_bn
        t_conv5 = self.list_tk['conv5'](v_conv5,'enc',self.epoch,training)
        v_conv5_dec = self.list_tk['conv5'](t_conv5,'dec',self.epoch,training)
        a_conv5 = v_conv5_dec
    else:
        a_conv5 = tf.nn.relu(s_conv5_bn)

    if training:
        a_conv5 = self.dropout_conv2(a_conv5,training=training)
    s_conv5_1 = self.list_layer['conv5_1'](a_conv5)
    if self.f_skip_bn:
        s_conv5_1_bn = s_conv5_1
    else:
        s_conv5_1_bn = self.list_layer['conv5_1_bn'](s_conv5_1,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv5_1 = s_conv5_1_bn
        t_conv5_1 = self.list_tk['conv5_1'](v_conv5_1,'enc',self.epoch,training)
        v_conv5_1_dec = self.list_tk['conv5_1'](t_conv5_1,'dec',self.epoch,training)
        a_conv5_1 = v_conv5_1_dec
    else:
        a_conv5_1 = tf.nn.relu(s_conv5_1_bn)

    if training:
        a_conv5_1 = self.dropout_conv2(a_conv5_1,training=training)
    s_conv5_2 = self.list_layer['conv5_2'](a_conv5_1)
    if self.f_skip_bn:
        s_conv5_2_bn = s_conv5_2
    else:
        s_conv5_2_bn = self.list_layer['conv5_2_bn'](s_conv5_2,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_conv5_2 = s_conv5_2_bn
        t_conv5_2 = self.list_tk['conv5_2'](v_conv5_2,'enc',self.epoch,training)
        v_conv5_2_dec = self.list_tk['conv5_2'](t_conv5_2,'dec',self.epoch,training)
        a_conv5_2 = v_conv5_2_dec
    else:
        a_conv5_2 = tf.nn.relu(s_conv5_2_bn)

    #print(tf.reduce_max(a_conv5_2))
    p_conv5_2 = self.pool2d(a_conv5_2)

    s_flat = tf.compat.v1.layers.flatten(p_conv5_2)

    if training:
        s_flat = self.dropout(s_flat,training=training)

    s_fc1 = self.list_layer['fc1'](s_flat)
    if self.f_skip_bn:
        s_fc1_bn = s_fc1
    else:
        s_fc1_bn = self.list_layer['fc1_bn'](s_fc1,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_fc1 = s_fc1_bn
        t_fc1 = self.list_tk['fc1'](v_fc1,'enc',self.epoch,training)
        v_fc1_dec = self.list_tk['fc1'](t_fc1,'dec',self.epoch,training)
        a_fc1 = v_fc1_dec
    else:
        a_fc1 = tf.nn.relu(s_fc1_bn)

    if training:
        a_fc1 = self.dropout(a_fc1,training=training)

    s_fc2 = self.list_layer['fc2'](a_fc1)
    if self.f_skip_bn:
        s_fc2_bn = s_fc2
    else:
        s_fc2_bn = self.list_layer['fc2_bn'](s_fc2,training=training)

    rand = tf.random.uniform(shape=(), minval=0, maxval=1)
    if training == False or ((training) and (tf.math.less(rand, pr_layer))):
        #if True:
        v_fc2 = s_fc2_bn
        t_fc2 = self.list_tk['fc2'](v_fc2,'enc',self.epoch,training)
        v_fc2_dec = self.list_tk['fc2'](t_fc2,'dec',self.epoch,training)
        a_fc2 = v_fc2_dec
    else:
        a_fc2 = tf.nn.relu(s_fc2_bn)
    if training:
        a_fc2 = self.dropout(a_fc2,training=training)


    s_fc3 = self.list_layer['fc3'](a_fc2)
    if self.f_skip_bn:
        s_fc3_bn = s_fc3
    else:
        if ('bn' in self.conf.model_name) or ('ro' in self.conf.model_name) :
            s_fc3_bn = self.list_layer['fc3_bn'](s_fc3,training=training)
        else:
            s_fc3_bn = s_fc3
    #a_fc3 = s_fc3_bn
    if 'ro' in self.conf.model_name:
        a_fc3 = tf.nn.relu(s_fc3_bn)
    else:
        a_fc3 = s_fc3_bn


    # print - activation histogram
    #if not self.f_1st_iter:
    if False:

        fig, axs = plt.subplots(4,5)
        axs=axs.ravel()

        list_hist_count=[]
        list_hist_bins=[]
        list_hist_bars=[]
        list_beta=[]
        list_x_M=[]
        list_x_m=[]

        #
        counts, bins, bars = axs[0].hist(x.numpy().flatten(),bins=1000)
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(0)
        list_x_M.append(0)
        list_x_m.append(0)

        #
        counts, bins, bars = axs[1].hist(s_conv1_bn.numpy().flatten(),bins=1000)
        layer_name = 'conv1'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[1].vlines(beta,0, np.max(counts), color='k')
        axs[1].vlines(x_m,0, np.max(counts), color='r')
        axs[1].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #
        counts, bins, bars = axs[2].hist(s_conv1_1_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv1_1'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[2].vlines(beta,0, np.max(counts), color='k')
        axs[2].vlines(x_m,0, np.max(counts), color='r')
        axs[2].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #
        counts, bins, bars = axs[3].hist(s_conv2_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv2'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[3].vlines(beta,0, np.max(counts), color='k')
        axs[3].vlines(x_m,0, np.max(counts), color='r')
        axs[3].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #
        counts, bins, bars = axs[4].hist(s_conv2_1_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv2_1'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[4].vlines(beta,0, np.max(counts), color='k')
        axs[4].vlines(x_m,0, np.max(counts), color='r')
        axs[4].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #
        counts, bins, bars = axs[5].hist(s_conv3_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv3'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[5].vlines(beta,0, np.max(counts), color='k')
        axs[5].vlines(x_m,0, np.max(counts), color='r')
        axs[5].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[6].hist(s_conv3_1_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv3_1'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[6].vlines(beta,0, np.max(counts), color='k')
        axs[6].vlines(x_m,0, np.max(counts), color='r')
        axs[6].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[7].hist(s_conv3_2_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv3_2'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[7].vlines(beta,0, np.max(counts), color='k')
        axs[7].vlines(x_m,0, np.max(counts), color='r')
        axs[7].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[8].hist(s_conv4_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv4'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[8].vlines(beta,0, np.max(counts), color='k')
        axs[8].vlines(x_m,0, np.max(counts), color='r')
        axs[8].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[9].hist(s_conv4_1_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv4_1'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[9].vlines(beta,0, np.max(counts), color='k')
        axs[9].vlines(x_m,0, np.max(counts), color='r')
        axs[9].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[10].hist(s_conv4_2_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv4_2'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[10].vlines(beta,0, np.max(counts), color='k')
        axs[10].vlines(x_m,0, np.max(counts), color='r')
        axs[10].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[11].hist(s_conv5_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv5'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[11].vlines(beta,0, np.max(counts), color='k')
        axs[11].vlines(x_m,0, np.max(counts), color='r')
        axs[11].vlines(x_M,0, np.max(counts), color='m')

        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #

        counts, bins, bars = axs[12].hist(s_conv5_1_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv5_1'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[12].vlines(beta,0, np.max(counts), color='k')
        axs[12].vlines(x_m,0, np.max(counts), color='r')
        axs[12].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[13].hist(s_conv5_2_bn.numpy().flatten(),bins=1000)

        layer_name = 'conv5_2'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[13].vlines(beta,0, np.max(counts), color='k')
        axs[13].vlines(x_m,0, np.max(counts), color='r')
        axs[13].vlines(x_M,0, np.max(counts), color='m')

        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #

        counts, bins, bars = axs[14].hist(s_fc1_bn.numpy().flatten(),bins=1000)

        layer_name = 'fc1'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[14].vlines(beta,0, np.max(counts), color='k')
        axs[14].vlines(x_m,0, np.max(counts), color='r')
        axs[14].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[15].hist(s_fc2_bn.numpy().flatten(),bins=1000)

        layer_name = 'fc2'
        beta = tf.math.reduce_mean(self.list_layer[layer_name+'_bn'].beta)
        x_M = tf.math.exp(tf.math.divide(self.list_tk[layer_name].td,self.list_tk[layer_name].tc))
        x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk[layer_name].tc)))
        axs[15].vlines(beta,0, np.max(counts), color='k')
        axs[15].vlines(x_m,0, np.max(counts), color='r')
        axs[15].vlines(x_M,0, np.max(counts), color='m')
        list_hist_count.append(counts)
        list_hist_bins.append(bins)
        list_hist_bars.append(bars)
        list_beta.append(beta.numpy())
        list_x_M.append(x_M.numpy()[0])
        list_x_m.append(x_m.numpy()[0])

        #


        counts, bins, bars = axs[16].hist(s_fc3_bn.numpy().flatten(),bins=1000)



        #
        #output_xlsx_name='bn_act_hist_SM-2.xlsx'
        #output_xlsx_name='bn_act_hist_SR.xlsx'
        #output_xlsx_name='bn_act_hist_TR.xlsx'
        output_xlsx_name='bn_act_hist_TB.xlsx'
        df=pd.DataFrame(list_hist_count).T
        #            #df=pd.DataFrame({'loss_prec': list_loss_prec, 'loss_min': list_loss_min, 'loss_max': list_loss_max})
        df.to_excel(output_xlsx_name,sheet_name='count')

        with pd.ExcelWriter(output_xlsx_name,mode='a') as writer:
            df=pd.DataFrame(list_hist_bins).T
            df.to_excel(writer,sheet_name='bins')

            df=pd.DataFrame(list_hist_bars).T
            df.to_excel(writer,sheet_name='bars')

            print(list_beta)
            df=pd.DataFrame(list_beta).T
            df.to_excel(writer,sheet_name='beta')

            df=pd.DataFrame(list_x_M).T
            df.to_excel(writer,sheet_name='x_M')

            df=pd.DataFrame(list_x_m).T
            df.to_excel(writer,sheet_name='x_m')


        # for data write
        #            #
        ##            col_x = x.numpy().flatten()
        ##            col_conv1_bn   = s_conv1_bn.numpy().flatten()
        ##            col_conv1_1_bn = s_conv1_1_bn.numpy().flatten()
        ##            col_conv2_bn   = s_conv2_bn.numpy().flatten()
        ##            col_conv2_1_bn = s_conv2_1_bn.numpy().flatten()
        ##            col_conv2_2_bn = s_conv2_2_bn.numpy().flatten()
        ##            col_conv3_bn   = s_conv3_bn.numpy().flatten()
        ##            col_conv3_1_bn = s_conv3_1_bn.numpy().flatten()
        ##            col_conv3_2_bn = s_conv3_2_bn.numpy().flatten()
        ##            col_conv4_bn   = s_conv4_bn.numpy().flatten()
        ##            col_conv4_1_bn = s_conv4_1_bn.numpy().flatten()
        ##            col_conv4_2_bn = s_conv4_2_bn.numpy().flatten()
        ##            col_conv5_bn   = s_conv5_bn.numpy().flatten()
        ##            col_conv5_1_bn = s_conv5_1_bn.numpy().flatten()
        ##            col_conv5_2_bn = s_conv5_2_bn.numpy().flatten()
        ##            col_fc1_bn     = s_fc1_bn.numpy().flatten()
        ##            col_fc2_bn     = s_fc2_bn.numpy().flatten()
        ##            col_fc3_bn     = s_fc3_bn.numpy().flatten()
        #
        #            #
        #            list_df=[]
        #            list_df.append(x.numpy().flatten())
        #            list_df.append(s_conv1_bn.numpy().flatten())
        #            list_df.append(s_conv1_1_bn.numpy().flatten())
        #
        #            df=pd.DataFrame(list_df)
        #            #df=pd.DataFrame({'loss_prec': list_loss_prec, 'loss_min': list_loss_min, 'loss_max': list_loss_max})
        #            df.to_excel('test.xlsx')
        #
        #            print(n)
        #            print(bins)
        #            print(patches)
        #
        #            print(tf.math.reduce_mean(self.list_layer['conv1_bn'].gamma))
        #            print(tf.math.reduce_mean(self.list_layer['conv1_bn'].beta))
        #            print(self.list_tk['conv1'].tc)
        #            print(self.list_tk['conv1'].td)
        #            x_M = tf.math.exp(tf.math.divide(self.list_tk['conv1'].td,self.list_tk['conv1'].tc))
        #            x_m = tf.math.multiply(x_M,tf.math.exp(tf.math.divide(-self.conf.time_window,self.list_tk['conv1'].tc)))
        #            print(x_M)
        #            print(x_m)

        plt.show()

        assert False



    a_out = a_fc3

    if self.f_1st_iter and self.conf.nn_mode=='ANN':
        print('1st iter')
        self.f_1st_iter = False
        self.f_skip_bn = (not self.f_1st_iter) and (self.conf.f_fused_bn)


    if not self.f_1st_iter and self.en_opt_time_const_T2FSNN:
        print("training time constant for temporal coding in SNN")

        self.dnn_act_list['in'] = a_in
        self.dnn_act_list['conv1']   = a_conv1
        self.dnn_act_list['conv1_1'] = a_conv1_1

        self.dnn_act_list['conv2']   = a_conv2
        self.dnn_act_list['conv2_1'] = a_conv2_1

        self.dnn_act_list['conv3']   = a_conv3
        self.dnn_act_list['conv3_1'] = a_conv3_1
        self.dnn_act_list['conv3_2'] = a_conv3_2

        self.dnn_act_list['conv4']   = a_conv4
        self.dnn_act_list['conv4_1'] = a_conv4_1
        self.dnn_act_list['conv4_2'] = a_conv4_2

        self.dnn_act_list['conv5']   = a_conv5
        self.dnn_act_list['conv5_1'] = a_conv5_1
        self.dnn_act_list['conv5_2'] = a_conv5_2

        self.dnn_act_list['fc1'] = a_fc1
        self.dnn_act_list['fc2'] = a_fc2
        self.dnn_act_list['fc3'] = a_fc3

    return a_out


