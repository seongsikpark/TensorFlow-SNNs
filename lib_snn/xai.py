import collections

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf


# OLD

#
# based on TF Tutorials
#@tf.function
def integrated_gradients(model,
                         baseline,
                         images,
                         target_class_idxs,
                         m_steps=50,
                         batch_size=51):

    # 1. Generate alphas
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients
    #gradient_batches = collections.OrderedDict()
    #for l in model.layers_w_kernel:
    #    gradient_batches[l.name] = tf.TensorArray(tf.float32, size=m_steps+1)
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           images=images,
                                                           alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs
        gradient_batch = compute_gradients(model = model,
                                           images=interpolated_path_input_batch,
                                           target_class_idxs=target_class_idxs)
        #for l in model.layers_w_kernel:
        #    # Write batch indices and gradients to extend Tensor Array
        #    gradient_batches[l.name] = gradient_batches[l.name].scatter(tf.range(from_,to), gradient_batch[l.name])
        gradient_batches.append(gradient_batch)

    # Stack path gradients together row-wise into single tensor
    #total_gradients = collections.OrderedDict()
    #for l in model.layers_w_kernel:
    #    total_gradients[l.name] = gradient_batches[l.name].stack()
    total_gradients = tf.concat(gradient_batches,axis=0)

    # 4. Integral approximation through averaging gradient
    avg_gradients = integral_approximation(model=model,gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input
    #scaled_integrated_gradients = collections.OrderedDict()
    #for l in model.layers_w_kernel:
    #    baseline_act = tf.zeros(l.record_output.shape)
    #    #scaled_integrated_gradients[l.name] = (images - baseline) *avg_gradients[l.name]
    #    scaled_integrated_gradients[l.name] = (l.record_output- baseline_act) *avg_gradients[l.name]
    integrated_gradients = (images-baseline)*avg_gradients

    # visualization gradient saturation
    preds=model(interpolated_path_input_batch)
    visual_gradient_saturation(alphas, preds, target_class_idxs, total_gradients)



    return integrated_gradients

    #integrated_gradients = (images - baseline) *avg_gradients

    #return integrated_gradients
    #return scaled_integrated_gradients


#
def interpolate_images(baseline,
                       images,
                       alphas):
    # only one image
    if True:
    #if False:
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(images, axis=0)
        delta = input_x - baseline_x
        interpolated_images = baseline_x + alphas_x*delta

    if False:
    #if True:
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(images, axis=0)
        delta = input_x - baseline_x
        interpolated_images = baseline_x + alphas_x * delta

    return interpolated_images

#
def compute_gradients(model, images, target_class_idxs):

    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idxs]

    return tape.gradient(probs, images)

    # maybe old
    if False:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(images)
            probs = model(images)
            top_class = tf.gather(probs, target_class_idxs)
            #probs = tf.gather(probs, target_class_idxs)

            grads_pred_act = collections.OrderedDict()

            for l in model.layers_w_kernel:
                act = l.record_output
                grads_pred_act[l.name] = tape.gradient(top_class, act)
                # grad * act
                grads_pred_act[l.name] = grads_pred_act[l.name]*act

        #print(act)
        #print(grads_pred_act)

        #assert False

        #return tape.gradient(probs, images)
        return grads_pred_act

#
def integral_approximation(model,gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients

    # old
    if False:
        # riemann_trapezoidal
        integrated_gradients = collections.OrderedDict()
        for l in model.layers_w_kernel:
            gradients_l = gradients[l.name]
            grads = (gradients_l[:-1] + gradients_l[1:]) / tf.constant(2.0)
            integrated_gradients[l.name] = tf.math.reduce_mean(grads, axis=0)

        return integrated_gradients


# visualization integrated gradients
#def plot_img_attributions(baseline,
                          #image,
                          #target_class_idx,
                          #m_steps=50,
                          #cmap=None,
                          #overlay_alpha=0.4):
def plot_image_attributions(baseline,
                            image,
                            attributions,
                            cmap=plt.cm.inferno,
                            overlay_alpha=0.4):
    #attributions = integrated_gradients
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions),axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8,8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()
    return fig


def visual_gradient_saturation(alphas, preds, class_idx, path_gradients):
    pred_proba = tf.nn.softmax(preds,axis=-1)[:,class_idx]

    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    ax1.plot(alphas, pred_proba)
    ax1.set_title('Target class predicted probability over alpha')
    ax1.set_ylabel('model p(target class)')
    ax1.set_xlabel('alpha')
    ax1.set_ylim([0,1])

    ax2 = plt.subplot(1,2,2)
    # Average across interpolation steps
    average_grads = tf.reduce_mean(path_gradients, axis=[1,2,3])
    # Normalize gradients to 0 to 1 scale. E.g. (x-min(x))/(max(x)-min(x))
    average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
    ax2.plot(alphas, average_grads_norm)
    ax2.set_title('Average pixel gradients (normalized) over alpha')
    ax2.set_ylabel('Average pixel gradients')
    ax2.set_xlabel('Alphas')
    ax2.set_ylim([0,1])
