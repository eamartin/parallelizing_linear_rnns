#! /bin/sh
rm -rf lib/

mkdir lib
nvcc -c linear_recurrence.cu -o lib/linear_recurrence.o -O3 --compiler-options '-fPIC'
nvcc lib/linear_recurrence.o -shared -o lib/liblinear_recurrence.so --compiler-options '-fPIC'

# building tensorflow op
export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared -o lib/tf_linear_recurrence.so linear_recurrent_net/tensorflow_binding/linear_recurrence_op.cpp lib/linear_recurrence.o -O3 -I $TF_INC -fPIC -lcudart
