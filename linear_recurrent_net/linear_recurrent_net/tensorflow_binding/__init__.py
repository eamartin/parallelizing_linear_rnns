import os
import tensorflow as tf
from tensorflow.python.framework import ops

dir = os.path.dirname(os.path.abspath(__file__))
_lr_module = tf.load_op_library('%s/../../lib/tf_linear_recurrence.so' % dir)

def linear_recurrence(decays, impulses, initial_state=None):
    '''
    Compute r[i] = decays[i] * r[i - 1] + impulses[i] with r[0] = initial_state.

    decays and impulses must have the same shape and are [n_steps, ...].
    initial_state must be None (to zero initialize) or [...]
    '''

    if initial_state is None:
        initial_state = tf.zeros_like(impulses[0, :])

    shape = tf.shape(decays)
    rank = shape.get_shape()[0].value
    if rank > 2:
        tail = tf.reduce_prod(shape[1:])
        decays = tf.reshape(decays, [shape[0], tail])
        impulses = tf.reshape(impulses, [shape[0], tail])
        initial_state = tf.reshape(initial_state, [tail])

    resp = _lr_module.linear_recurrence(decays, impulses, initial_state)

    if rank > 2:
        resp = tf.reshape(resp, shape)
    return resp

@ops.RegisterGradient("LinearRecurrence")
def _linear_recurrence_grad(op, dl_dresp):
    decays = op.inputs[0]
    impulses = op.inputs[1]
    initial_state = op.inputs[2]

    n_steps = tf.shape(impulses)[0]

    # forwards goes from h_0 to h_{T-1}
    forwards_tail = linear_recurrence(decays, impulses, initial_state)[:-1, :]
    forwards = tf.concat([tf.expand_dims(initial_state, 0), forwards_tail],
                         axis=0)

    reverse = lambda x: tf.reverse(x, axis=[0])

    # recur on
    # decays from T, T-1, ..., 2
    # output gradients from T-1, T-2, ..., 1
    dl_dh_head = reverse(
        linear_recurrence(
            reverse(decays)[:-1, :],
            reverse(dl_dresp)[1:, :],
            dl_dresp[-1, :],
        )
    )

    dl_dh = tf.concat([dl_dh_head, dl_dresp[-1:, :]], axis=0)

    dl_dinit = decays[0, :] * dl_dh[0, :]
    dl_dimpulses = dl_dh
    dl_ddecays = dl_dh * forwards

    return [dl_ddecays, dl_dimpulses, dl_dinit]
