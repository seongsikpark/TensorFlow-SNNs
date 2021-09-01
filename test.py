
assert False

#def run(model_name,dataset_name,num_class,input_size,)

if dataset_name == 'ImageNet':

    pretrained_model.compile(optimizer='adam',
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             #metrics=['accuracy'])
                             #metrics=[tf.keras.metrics.sparse_top_k_categorical_accuracy])
                             metrics=[tf.keras.metrics.categorical_accuracy, \
                                      tf.keras.metrics.top_k_categorical_accuracy])

    # Preprocess input
    #ds=ds.map(resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds=valid_ds.map(daug.resize_with_crop,num_parallel_calls=NUM_PARALLEL_CALL)
    #valid_ds=valid_ds.map(eager_resize_with_crop,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_ds=valid_ds.batch(batch_size_inference)
    #valid_ds=valid_ds.batch(250)
    #valid_ds=valid_ds.batch(2)
    #valid_ds=valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds=valid_ds.prefetch(NUM_PARALLEL_CALL)

    #ds=ds.take(1)

    result = pretrained_model.evaluate(valid_ds)
