import tensorflow as tf

def conv2(inp_tensor, num_filters, size=[3,1], stride = (1,1), padding_scheme = 'valid', \
    activation_fn=tf.nn.relu, k_init=tf.contrib.layers.xavier_initializer()):

    cnv =  tf.layers.conv2d(inp_tensor, num_filters, size, stride , padding_scheme,\
    activation = activation_fn, kernel_initializer = k_init)
    return cnv
    # return tf.nn.dropout(cnv, 0.5)

def mp(inp_tensor):
    return tf.layers.max_pooling2d(inp_tensor,[2,1],(2,1))