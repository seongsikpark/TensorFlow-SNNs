
'''

'''


from absl import flags
conf = flags.FLAGS

#conf.train_epoch = 100
#conf.num_train_data = 10000

conf.nn_mode = 'SNN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'

conf.n_type = 'IF'
