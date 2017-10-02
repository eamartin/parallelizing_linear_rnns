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
             name='fc'):
    n_dims = X.get_shape()[-1].value
    with vscope(name):
        W = tf.get_variable('W', [n_dims, hidden_size])

        if use_bias:
            b = tf.get_variable('b', [hidden_size])
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
                          nonlin=tf.identity)

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
        preact = fc_layer(X, 3 * size, nonlin=tf.identity)
        x_tilde, f_pre, r_pre = tf.split(preact, 3, len(preact.shape) - 1)

        f = tf.sigmoid(f_pre)
        r = tf.sigmoid(r_pre)

        c = linear_recurrence(f, (1 - f) * x_tilde)
        h = r * c + (1 - r) * X
        return h
