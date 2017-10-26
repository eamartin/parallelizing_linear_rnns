import tensorflow as tf
from linear_recurrent_net.tensorflow_binding import linear_recurrence

def vscope(name):
    return tf.variable_scope(None, default_name=name)

# contracts on the innermost dimension
def matmul(X, W):
    res = tf.tensordot(X, W, [[-1], [0]])
    res.set_shape(X.get_shape().as_list()[:-1] +
                  W.get_shape().as_list()[1:])
    return res

def embedding_layer(X, size, dims, name='embedding'):
    with vscope(name):
        W = tf.get_variable('W', [dims, size])
        return tf.nn.embedding_lookup(W, X)

def fc_layer(X, hidden_size, nonlin=tf.nn.elu,
             use_bias=True, use_layer_norm=False, ln_eps=1e-3,
             name='fc', sn=0.05, forget_bias=5.0):
    n_dims = X.get_shape()[-1].value
    with vscope(name):
        W = tf.get_variable('W', initializer=tf.random_uniform([n_dims, hidden_size], maxval=sn, minval=-sn))

        if use_bias and name == 'pre_fc':
            b = tf.get_variable('b', initializer=tf.concat([tf.constant(forget_bias, shape=[hidden_size/4]),
                                                            tf.zeros([3*(hidden_size/4)])],axis=0))
        elif use_bias and name == 'sru_pre':
            b = tf.get_variable('b', initializer=tf.concat([tf.zeros([(hidden_size/3)]), 
                                                             tf.constant(forget_bias, shape=[hidden_size/3]),
                                                             tf.zeros([(hidden_size/3)])],axis=0))            
        elif use_bias:
            b = tf.get_variable('b', initializer=tf.zeros([hidden_size]))
        else:
            b = 0

        prod = matmul(X, W)
        if use_layer_norm:
            idx = ([None] * (len(X.shape) - 1)) + [slice(None)]
            g = tf.get_variable('g', [hidden_size])[idx]

            mu, sigma = tf.nn.moments(prod, [-1], keep_dims=True)
            prod = g * (prod - mu) / (sigma + ln_eps)

        return nonlin(prod + b)

def gilr_layer(X, hidden_size, nonlin=tf.nn.elu,
               name='gilr'):
    """
    g_t = sigmoid(Ux_t + b)
    h_t = g_t h_{t-1} + (1-g_t) f(Vx_t + c)
    """
    with vscope(name):
        n_dims = X.get_shape()[-1].value
        act = fc_layer(X, 2 * hidden_size, nonlin=tf.identity)
        gate, impulse = tf.split(act, 2, len(act.shape) - 1)
        gate = tf.sigmoid(gate)
        impulse = nonlin(impulse)
        return linear_recurrence(gate, (1-gate) * impulse)

def linear_surrogate_lstm(X, hidden_size, name='lin_sur_lstm'):
    with vscope(name):
        # 2 * hidden_size * n_dims params
        h_tilde = gilr_layer(X, hidden_size, nonlin=tf.tanh)

        # 4 * hidden_size * (hidden_size + n_dims)
        preact = fc_layer(tf.concat([h_tilde, X], axis=-1), 4 * hidden_size,
                          nonlin=tf.identity, name='pre_fc')

        f, i, o, z = tf.split(preact, 4, len(preact.shape) - 1)

        f = tf.sigmoid(f)
        i = tf.sigmoid(i)
        o = tf.sigmoid(o)
        z = tf.tanh(z)

        c = linear_recurrence(f, i * z)
        h = o * c
        return h

def SRU(X, name='SRU'):
    size = X.get_shape()[-1].value
    with vscope(name):
        preact = fc_layer(X, 3 * size, nonlin=tf.identity, name='sru_pre')
        x_tilde, f_pre, r_pre = tf.split(preact, 3, len(preact.shape) - 1)
        
        f = tf.sigmoid(f_pre)
        r = tf.sigmoid(r_pre)
        
        c = linear_recurrence(f, (1 - f) * x_tilde)
        h = r * c + (1 - r) * X
        return h


def QRNN(X, n, name='qrnn'):
    size = X.get_shape()[-1].value
    length = X.get_shape()[0].value
    bs = X.get_shape()[1].value
    with vscope(name):
        stack_list = []
        for m in range(1, n-1):
            stack_list.append(tf.slice(tf.pad(X, [[m,0], [0,0], [0,0]]),
                                       [0,0,0], [length, bs, size]))
        X_stacked = tf.concat([X] + stack_list, axis=-1)

        preact = fc_layer(X_stacked, 3 * n * size, nonlin=tf.identity, name='qrnn_pre')

        z, f, o = tf.split(preact, 3, len(preact.shape) - 1)

        z = tf.tanh(tf.add_n(tf.split(z, n, len(preact.shape) - 1)))
        f = tf.sigmoid(tf.add_n(tf.split(f, n, len(preact.shape) - 1)))
        o = tf.sigmoid(tf.add_n(tf.split(o, n, len(preact.shape) - 1)))

        c = linear_recurrence(f, (1 - f) * z)
        h = o * c
        return h    

def s_gilr_layer(X, hidden_size, nonlin=tf.nn.elu,
               name='gilr'):
    """
    g_t = sigmoid(Ux_t + b)
    h_t = g_t h_{t-1} + (1-g_t) f(Vx_t + c)
    """
    with vscope(name):
        n_dims = X.get_shape()[-1].value
        act = fc_layer(X, 2 * hidden_size, nonlin=tf.identity)
        gate, impulse = tf.split(act, 2, len(act.shape) - 1)
        gate = tf.sigmoid(gate)
        impulse = nonlin(impulse)
        return linear_recurrence(gate, (1-gate) * impulse, serial=True)

def s_linear_surrogate_lstm(X, hidden_size, name='lin_sur_lstm'):
    with vscope(name):
        # 2 * hidden_size * n_dims params
        h_tilde = gilr_layer(X, hidden_size, nonlin=tf.tanh)

        # 4 * hidden_size * (hidden_size + n_dims)
        preact = fc_layer(tf.concat([h_tilde, X], axis=-1), 4 * hidden_size,
                          nonlin=tf.identity, name='pre_fc')

        f, i, o, z = tf.split(preact, 4, len(preact.shape) - 1)

        f = tf.sigmoid(f)
        i = tf.sigmoid(i)
        o = tf.sigmoid(o)
        z = tf.tanh(z)

        c = linear_recurrence(f, i * z, serial=True)
        h = o * c
        return h

def s_SRU(X, name='SRU'):
    size = X.get_shape()[-1].value
    with vscope(name):
        preact = fc_layer(X, 3 * size, nonlin=tf.identity, name='sru_pre')
        x_tilde, f_pre, r_pre = tf.split(preact, 3, len(preact.shape) - 1)
        
        f = tf.sigmoid(f_pre)
        r = tf.sigmoid(r_pre)
        
        c = linear_recurrence(f, (1 - f) * x_tilde, serial=True)
        h = r * c + (1 - r) * X
        return h
    
def s_QRNN(X, n, name='qrnn'):
    size = X.get_shape()[-1].value
    length = X.get_shape()[0].value
    bs = X.get_shape()[1].value
    with vscope(name):
        stack_list = []
        for m in range(1, n-1):
            stack_list.append(tf.slice(tf.pad(X, [[m,0], [0,0], [0,0]]),
                                       [0,0,0], [length, bs, size]))
        X_stacked = tf.concat([X] + stack_list, axis=-1)

        preact = fc_layer(X_stacked, 3 * n * size, nonlin=tf.identity, name='qrnn_pre')

        z, f, o = tf.split(preact, 3, len(preact.shape) - 1)

        z = tf.tanh(tf.add_n(tf.split(z, n, len(preact.shape) - 1)))
        f = tf.sigmoid(tf.add_n(tf.split(f, n, len(preact.shape) - 1)))
        o = tf.sigmoid(tf.add_n(tf.split(o, n, len(preact.shape) - 1)))

        c = linear_recurrence(f, (1 - f) * z, serial=True)
        h = o * c
        return h


def linear_surrogate_lstm_cpu(X, hidden_size, name='lin_sur_lstm'):
    with vscope(name):
        # 2 * hidden_size * n_dims params
        h_tilde = gilr_layer(X, hidden_size, nonlin=tf.tanh)

        # 4 * hidden_size * (hidden_size + n_dims)
        preact = fc_layer(tf.concat([h_tilde, X], axis=-1), 4 * hidden_size,
                          nonlin=tf.identity, name='pre_fc')

        f, i, o, z = tf.split(preact, 4, len(preact.shape) - 1)

        f = tf.sigmoid(f)
        i = tf.sigmoid(i)
        o = tf.sigmoid(o)
        z = tf.tanh(z)

        c = linear_recurrence_cpu(f, i * z)
        h = o * c
        return h

def gilr_layer_cpu(X, hidden_size, nonlin=tf.nn.elu,
               name='gilr'):
    """
    g_t = sigmoid(Ux_t + b)
    h_t = g_t h_{t-1} + (1-g_t) f(Vx_t + c)
    """
    with vscope(name):
        n_dims = X.get_shape()[-1].value
        act = fc_layer(X, 2 * hidden_size, nonlin=tf.identity)
        gate, impulse = tf.split(act, 2, len(act.shape) - 1)
        gate = tf.sigmoid(gate)
        impulse = nonlin(impulse)
        return s_linear_recurrence_cpu(gate, (1-gate) * impulse)

def linear_recurrence_cpu(f, b):
    """Compute the linear recurrence using native tf operations
    so that we evaluate without a GPU. We evaluate the recurrence
    which is stepwise h_t = f * h_{t-1} + b, returning all h."""
    fs = tf.unstack(f, axis=0)
    bs = tf.unstack(b, axis=0)
    h = tf.identity(b)

    hs = [bs[0]]
    for index in range(1, len(bs)):
        print fs[index], bs[index]
        to_append = tf.add(tf.multiply(fs[index], hs[index-1]), bs[index])
        hs.append(to_append)
    return tf.stack(hs)

