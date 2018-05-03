import tensorflow as tf

def conv2(inp_tensor, dropoutProb, num_filters, size=[2,1], stride = (2,1), padding_scheme = 'valid', \
    activation_fn=tf.nn.relu, k_init=tf.contrib.layers.xavier_initializer()):

    cnv =  tf.layers.conv2d(inp_tensor, num_filters, size, stride , padding_scheme,\
    activation = activation_fn, kernel_initializer = k_init)
    return cnv
    # print(dropoutProb.shape)
    # return tf.nn.dropout(cnv, dropoutProb)
    
