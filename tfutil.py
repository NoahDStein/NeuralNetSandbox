import time
import os
import numpy
import tensorflow as tf


def listify(x):
    try:
        len(x)
    except TypeError:
        return [x]
    return x


# Not averaged over examples
def multi_class_hinge_loss(inner_products, labels, power=1):
    membership_indicator = tf.one_hot(labels, tf.shape(inner_products)[1], on_value=1.0, off_value=-1.0)
    hinge_loss = tf.nn.relu(1.0 - membership_indicator * inner_products)
    if power !=1:
        if power == 2:
            hinge_loss = tf.square(hinge_loss)
        else:
            hinge_loss = tf.pow(hinge_loss, power)
    return tf.reduce_sum(hinge_loss, reduction_indices=[1])

def log(s):
    print('[%s] ' % time.asctime() + s)


def restore_latest(saver, sess, path, suffix=''):
    dated_files = [(os.path.getmtime(path + '/' + fn), os.path.basename(fn)) for fn in os.listdir(path) if
                   fn.startswith('save') and fn.endswith(suffix) and os.path.splitext(fn)[1] == '']
    dated_files.sort()
    dated_files.reverse()
    newest = dated_files[0][1]
    log('restoring %s updated at %s' % (dated_files[0][1], time.ctime(dated_files[0][0])))
    saver.restore(sess, path + '/' + newest)


def modified_dynamic_shape(tensor, new_shape):
    return tuple(new_dim or tf.shape(tensor)[i] for i, new_dim in enumerate(new_shape))


def modified_static_shape(tensor, new_shape):
    return tuple(new_dim or tensor.get_shape().as_list()[i] for i, new_dim in enumerate(new_shape))


def roll0d(tensor, shift):  # roll tensor along 0th dimension
    n = tensor.get_shape()[0]
    if shift >= 0:
        z = tf.concat(concat_dim=0, values=[tf.gather(tensor, indices=tf.range(n-shift, n)), \
                                           tf.gather(tensor, indices=tf.range(n-shift))])
    else:
        z = tf.concat(concat_dim=0, values=[tf.gather(tensor, indices=tf.range(-shift, n)), \
                                           tf.gather(tensor, indices=tf.range(-shift))])
    return z


def tensor_roll_scalar(tensor, shift, axis, ndim):  # roll a tensor by a scalar argument
    dims = numpy.arange(ndim)
    if axis != 0:
        dims[axis], dims[0] = dims[0], dims[axis]
        permuted_tensor = tf.transpose(tensor, perm=dims)
    else:
        permuted_tensor = tensor
    return tf.transpose(roll0d(permuted_tensor, shift), perm=dims)


def quantizer(val, lower, upper, levels):
    normalized = (tf.clip_by_value(val, lower, upper) - lower)/(upper - lower + 1e-6)
    return tf.cast(tf.floor(normalized*levels), tf.int64)


def dequantizer(val, lower, upper, levels):
    return lower + (upper - lower) * tf.cast(val, tf.float32) / tf.cast(levels - 1, tf.float32)


def draw_on(canvas, to_draw, color):
    to_draw = tf.expand_dims(to_draw, 3)
    color = tf.constant(color, tf.float32)[None, None, None, :]
    return canvas*(1.0 - to_draw) + color*to_draw


def crappy_plot(val, levels):
    x_len = val.get_shape().as_list()[1]
    left_val = tf.concat(1, (val[:, 0:1], val[:, 0:x_len - 1]))
    right_val = tf.concat(1, (val[:, 1:], val[:, x_len - 1:]))

    left_mean = (val + left_val) // 2
    right_mean = (val + right_val) // 2
    low_val = tf.minimum(tf.minimum(left_mean, right_mean), val)
    high_val = tf.maximum(tf.maximum(left_mean, right_mean), val + 1)
    return tf.cumsum(tf.one_hot(low_val, levels, axis=1) - tf.one_hot(high_val, levels, axis=1), axis=1)


def queue_append_and_update(axis, old_contents, contents_to_append):
    ndims = old_contents.get_shape().ndims
    slice_begin = numpy.zeros(shape=(ndims,), dtype=numpy.int32)
    slice_begin[axis] = contents_to_append.get_shape().as_list()[axis]
    slice_size = -numpy.ones(shape=(ndims,), dtype=numpy.int32)
    concatenated_contents = tf.concat(axis, (old_contents[:contents_to_append.get_shape().as_list()[0], ...], contents_to_append))
    paddings = [[0, 0]] * ndims
    paddings[0] = [0, old_contents.get_shape().as_list()[0] - contents_to_append.get_shape().as_list()[0]]
    updated_contents = tf.pad(tf.slice(concatenated_contents, slice_begin, slice_size), paddings)
    return concatenated_contents, updated_contents
