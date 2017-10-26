def plr_slr(bs_seq_len_list):
    """Given a list of pairs (batch size, seq_len), 
    calculate the throughput of an LS-LSTM, an SRU, a QRNN(2),
    and QRNN(10) using the parallel kernel as opposed to the serial
    one"""
    import tensorflow as tf
    import numpy as np
    import scipy.io.wavfile
    from tensorflow.contrib import rnn
    import math
    from layers_new import linear_surrogate_lstm
    from layers_new import s_linear_surrogate_lstm
    from layers_new import SRU
    from layers_new import s_SRU
    from layers_new import QRNN
    from layers_new import s_QRNN        
    import time
    import os
    import random

    throughput_list = []

    #TODO:
    #Make LS_LSTM with PLR
    #Make SRU with PLR
    #Make QRNN with PLR
    #Make LS_LSTM with SLR
    #Make SRU with SLR
    #Make QRNN with SLR
    

    for seq_len in seq_len_list:
        #First generate the LS-LSTM and work out the throughput
        tf.reset_default_graph()        
        n_hidden = 256
        n_classes = 2
        n_steps = seq_len
        batch_size = 65536 / seq_len
        bs = batch_size
        print "Batch size is {} and sequence length is {}".format(bs, seq_len)
        n_input = 24
        n_layers = 2
        forget_gate_init = 1.0                          # = 1/(n_in). We use uniform p(x)
        #Training Parameters
        sn = 1.0 / math.sqrt(n_hidden)
        learning_rate = 0.001
        training_iters = 5000000

        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_hidden, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')

        layer1 = linear_surrogate_lstm(x, n_hidden, name='ls-lstm')
        outputs = linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm2')    
        pred = tf.matmul(outputs[-1], W1) + b1
        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        ls_lstm_tp = (bs * n_steps) / np.mean(times)


        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_hidden, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = s_linear_surrogate_lstm(x, n_hidden, name='ls-lstm')
        output = s_linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm')        
        pred = tf.matmul(output[-1], W1) + b1

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        s_ls_lstm_tp = (bs * n_steps) / np.mean(times)


        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_input, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = SRU(x, name='SRU_1')
        output = SRU(layer1, name='SRU_2')
        pred = tf.matmul(output[-1], W1) + b1

        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_hidden, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = s_linear_surrogate_lstm(x, n_hidden, name='ls-lstm')
        output = s_linear_surrogate_lstm(layer1, n_hidden, name='ls-lstm')        
        pred = tf.matmul(output[-1], W1) + b1

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        s_ls_lstm_tp = (bs * n_steps) / np.mean(times)

        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_input, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = SRU(x, name='SRU_1')
        output = SRU(layer1, name='SRU_2')
        pred = tf.matmul(output[-1], W1) + b1        

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        sru_tp = (bs * n_steps) / np.mean(times)        


        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_input, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = s_SRU(x, name='s_SRU_1')
        output = s_SRU(layer1, name='s_SRU_2')
        pred = tf.matmul(output[-1], W1) + b1        

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        s_sru_tp = (bs * n_steps) / np.mean(times)
        

        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_input, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = QRNN(x, 2, name='QRNN_1')
        output = QRNN(layer1, 2, name='QRNN_2')
        pred = tf.matmul(output[-1], W1) + b1

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        qrnn_2_tp = (bs * n_steps) / np.mean(times)


        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_input, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = s_QRNN(x, 2, name='s_QRNN_3')
        output = s_QRNN(layer1, 2, name='s_QRNN_4')
        pred = tf.matmul(output[-1], W1) + b1

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        s_qrnn_2_tp = (bs * n_steps) / np.mean(times)
        print np.mean(times)
        print np.std(times)

        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_input, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = QRNN(x, 10, name='QRNN_2')
        output = QRNN(layer1, 10, name='QRNN_6')
        pred = tf.matmul(output[-1], W1) + b1

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        qrnn_10_tp = (bs * n_steps) / np.mean(times)


        tf.reset_default_graph()        
        x = tf.placeholder("float", [n_steps, batch_size, n_input])
        y = tf.placeholder("float", [batch_size, n_classes])
        tf.get_variable_scope().reuse == True
        W1 = tf.get_variable('W1', initializer=
                             tf.random_normal([n_input, n_classes]), dtype='float')
        b1 = tf.get_variable('b1', initializer=tf.zeros([n_classes]), dtype='float')
        layer1 = s_QRNN(x, 10, name='s_QRNN_7')
        output = s_QRNN(layer1, 10, name='s_QRNN_8')
        pred = tf.matmul(output[-1], W1) + b1

        #Evaluate network, run adam and clip gradients
        ################################################################################
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer_0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        raw_gradients, variables = zip(*optimizer_0.compute_gradients(cost))
        gradients = raw_gradients
        optimizer = optimizer_0.apply_gradients(zip(gradients, variables))
        init = tf.global_variables_initializer()

        #Initialise the model and evaluate
        step = 0
        times = []
        x_in = np.random.random((n_steps, batch_size, n_input))
        y_in = np.random.random((batch_size, n_classes))
        with tf.device("gpu:0"):
            with tf.Session() as sess:
                sess.run(init)
                while step < 10:
                    out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                    step += 1
                    if step != 0:
                        start = time.time()
                        out = sess.run(pred, feed_dict={x: x_in, y: y_in})
                        finish = time.time()
                        times.append(finish - start)
        s_qrnn_10_tp = (bs * n_steps) / np.mean(times)

        
        throughput_list.append([ls_lstm_tp, s_ls_lstm_tp, sru_tp,
                                s_sru_tp, qrnn_2_tp, s_qrnn_2_tp,
                                qrnn_10_tp, s_qrnn_10_tp])
    return throughput_list

if __name__ == "__main__":
    import numpy as np
    seq_len_list = [16 ** x for x in range(1, 5)]    
    out = plr_slr(seq_len_list)
    p_ls_lstm, s_ls_lstm, p_sru, s_sru, p_2_qrnn, s_2_qrnn, p_10_qrnn, s_10_qrnn = zip(*out)
    print np.array(p_ls_lstm) / np.array(s_ls_lstm)
    print np.array(p_sru) / np.array(s_sru) 
    print np.array(p_2_qrnn) / np.array(s_2_qrnn)
    print np.array(p_10_qrnn) / np.array(s_10_qrnn)     
    # in_list1 = [[1, x] for x in [2**z for z in range(8, 19-1)]]
    # in_list2 = [[2, x] for x in [2**z for z in range(8, 19-2)]]
    # in_list4 = [[4, x] for x in [2**z for z in range(8, 19-3)]]
    # in_list8 = [[8, x] for x in [2**z for z in range(8, 19-4)]]
    # in_list16 = [[16, x] for x in [2**z for z in range(8, 19-5)]]
    # in_list32 = [[32, x] for x in [2**z for z in range(8, 19-6)]]
    # in_list64 = [[64, x] for x in [2**z for z in range(8, 19-7)]]
    # in_list128 = [[128, x] for x in [2**z for z in range(8, 19-8)]]
    # in_list256 = [[256, x] for x in [2**z for z in range(8, 19-9)]]                                

    # in_list1.extend(in_list2)
    # in_list1.extend(in_list4)
    # in_list1.extend(in_list8)
    # in_list1.extend(in_list16)
    # in_list1.extend(in_list32)
    # in_list1.extend(in_list64)
    # in_list1.extend(in_list128)
    # in_list1.extend(in_list256)

    # out = random_test(in_list1)
    # print out
    # lstm_times, cudnn_times, speedups = zip(*out)
    
    
