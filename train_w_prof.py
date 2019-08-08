from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import functools

from tensorflow.python.platform import gfile
from tensorflow.python.eager import context
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import profile_context

#
def loss(predictions, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=labels))

#
def compute_accuracy(predictions, labels):
    return tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.argmax(predictions, axis=1, output_type=tf.int64),
                        tf.argmax(labels, axis=1, output_type=tf.int64)
                    ), dtype=tf.float32
                )
            ) / float(predictions.shape[0].value)

#@profile
def train_one_epoch(model, optimizer, dataset):
    tf.train.get_or_create_global_step()
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')

    def model_loss(labels, images):
        prediction = model(images, f_training=True)
        loss_value = loss(prediction, labels)
        avg_loss(loss_value)
        accuracy(tf.argmax(prediction,axis=1,output_type=tf.int64), tf.argmax(labels,axis=1,output_type=tf.int64))
        #tf.contrib.summary.scalar('loss', loss_value)
        #tf.contrib.summary.scalar('accuracy', compute_accuracy(prediction, labels))
        return loss_value

    #builder = tf.profiler.ProfileOptionBuilder
    #opts = builder(builder.time_and_memory()).order_by('micros').build()

    for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
        #print(batch)
        #print(images)
        #print(labels)
        #with tf.contrib.summary.record_summaries_every_n_global_steps(10):

        #with tf.contrib.tfprof.ProfileContext('./profile/train_'+conf.ann_model+'_'+conf.dataset) as pctx:
            #pctx.trace_next_step()
            #pctx.dump_next_step()
            #batch_model_loss = functools.partial(model_loss, labels, images)
            #optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())
            #pctx.profiler.profile_operations(options=opts)



        # profiler
#        with context.eager_mode():
#            outfile = os.path.join('./profile/train','dump')
#            opts = builder(builder.time_and_memory()).with_file_output(outfile).build()
#            context.enable_run_metadata()
#
#
#            batch_model_loss = functools.partial(model_loss, labels, images)
#            optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())
#
#            profiler = model_analyzer.Profiler()
#            profiler.add_step(0, context.export_run_metadata())
#            context.disable_run_metadata()
#            profiler.profile_operations(opts)
#
#            with gfile.Open(outfile, 'r') as f:
#                out_str = f.read()
#                print(out_str)


        batch_model_loss = functools.partial(model_loss, labels, images)
        optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())



        #batch_model_loss = functools.partial(model_loss, labels, images)
        #optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())


        #if batch==0:
        #    profiler.add_step(batch, run_meta)
        #    opts=option_builder.ProfileOptionBuilder.time_and_memory()
        #    profiler.profile_operations(options=opts)

        #profiler.advise()

    #print('Train set: Accuracy: %4f%%\n'%(100*accuracy.result()))
    return avg_loss.result(), 100*accuracy.result()


