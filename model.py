
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import feudal_batch_processor

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class SingleStepLSTM(object):

    def __init__(self,x,size,step_size):
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, 
                shape=[1, lstm.state_size.c],
                name='c_in')
        h_in = tf.placeholder(tf.float32, 
                shape=[1, lstm.state_size.h],
                name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs

class FeudalPolicy():
    """
    Policy of the Feudal network architecture.
    """

    def __init__(self, obs_space, act_space,global_step, config):
        self.global_step = global_step
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.k = config.k #Dimensionality of w
        self.g_dim = config.g_dim
        self.c = config.c
        self.batch_processor = feudal_batch_processor.FeudalBatchProcessor(self.c)
        self._build_model()

    def _build_model(self):
        """
        Builds the manager and worker models.
        """
        with tf.variable_scope('FeUdal'):
            self._build_placeholders()
            self._build_perception()
            self._build_manager()
            self._build_worker()
            self._build_loss()
            self.var_list = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        self.state_in = [self.worker_lstm.state_in[0], \
                         self.worker_lstm.state_in[1], \
                         self.manager_lstm.state_in[0], \
                         self.manager_lstm.state_in[1] \
                         ]
        self.state_out = [self.worker_lstm.state_out[0], \
                          self.worker_lstm.state_out[1], \
                          self.manager_lstm.state_out[0], \
                          self.manager_lstm.state_out[1] \
                          ]

    def _build_placeholders(self):
        #standard for all policies
        self.obs = tf.placeholder(tf.float32, [None] + list(self.obs_space))
        self.r = tf.placeholder(tf.float32,(None,))
        self.ac = tf.placeholder(tf.float32,(None,self.act_space))
        self.adv = tf.placeholder(tf.float32, [None]) #unused

        #specific to FeUdal
        self.prev_g = tf.placeholder(tf.float32, (None,None,self.g_dim))
        self.ri = tf.placeholder(tf.float32,(None,))
        self.s_diff = tf.placeholder(tf.float32,(None,self.g_dim))


    def _build_perception(self):
        conv1 = tf.layers.conv2d(inputs=self.obs,
                                 filters=16,
                                 kernel_size=[8, 8],
                                 activation=tf.nn.elu,
                                 strides=4)
        conv2 = tf.layers.conv2d(inputs=conv1,
                                 filters=32,
                                 kernel_size=[4,4],
                                 activation=tf.nn.elu,
                                 strides=2)

        flattened_filters = flatten(conv2)
        self.z = tf.layers.dense(inputs=flattened_filters, \
                                 units=256, \
                                 activation=tf.nn.elu)

    def _build_manager(self):
        with tf.variable_scope('manager'):
            # Calculate manager internal state
            self.s = tf.layers.dense(inputs=self.z, \
                                     units=self.g_dim, \
                                     activation=tf.nn.elu)

            # Calculate manager output g
            x = tf.expand_dims(self.s, [0])
            self.manager_lstm = SingleStepLSTM(x, \
                                               self.g_dim, \
                                               step_size=tf.shape(self.obs)[:1])
            g_hat = self.manager_lstm.output
            self.g = tf.nn.l2_normalize(g_hat, dim=1)

            self.manager_vf = self._build_value(g_hat)
            # self.manager_vf = tf.Print(self.manager_vf,[self.manager_vf])

    def _build_worker(self):
        with tf.variable_scope('worker'):
            num_acts = self.act_space

            # Calculate U
            self.worker_lstm = SingleStepLSTM(tf.expand_dims(self.z, [0]), \
                                              size=num_acts * self.k,
                                              step_size=tf.shape(self.obs)[:1])
            flat_logits = self.worker_lstm.output

            self.worker_vf = self._build_value(flat_logits)

            U = tf.reshape(flat_logits,[-1,num_acts,self.k])

            # Calculate w
            cut_g = tf.stop_gradient(self.g)
            cut_g = tf.expand_dims(cut_g, [1])
            gstack = tf.concat([self.prev_g,cut_g], axis=1)

            self.last_c_g = gstack[:,1:]
            # print self.last_c_g
            gsum = tf.reduce_sum(gstack, axis=1)
            phi = tf.get_variable("phi", (self.g_dim, self.k))
            w = tf.matmul(gsum,phi)
            w = tf.expand_dims(w,[2])
            # Calculate policy and sample
            logits = tf.reshape(tf.matmul(U,w),[-1,num_acts])
            self.pi = tf.nn.softmax(logits)
            self.log_pi = tf.nn.log_softmax(logits)
            self.sample = categorical_sample(
                tf.reshape(logits,[-1,num_acts]), num_acts)[0, :]

    def _build_value(self,input):
        with tf.variable_scope('VF'):
            hidden = tf.layers.dense(inputs=input, \
                                     units=self.config.vf_hidden_size, \
                                     activation=tf.nn.elu)

            w = tf.get_variable("weights", (self.config.vf_hidden_size, 1))
            return tf.matmul(hidden,w)

    def _build_loss(self):
        cutoff_vf_manager = tf.reshape(tf.stop_gradient(self.manager_vf),[-1])
        dot = tf.reduce_sum(tf.multiply(self.s_diff,self.g ),axis=1)
        gcut = tf.stop_gradient(self.g)
        mag = tf.norm(self.s_diff,axis=1)*tf.norm(gcut,axis=1)+.0001
        dcos = dot/mag
        manager_loss = -tf.reduce_sum((self.r-cutoff_vf_manager)*dcos)

        cutoff_vf_worker = tf.reshape(tf.stop_gradient(self.worker_vf),[-1])
        log_p = tf.reduce_sum(self.log_pi*self.ac,[1])
        worker_loss = (self.r + self.config.alpha*self.ri - cutoff_vf_worker)*log_p
        worker_loss = -tf.reduce_sum(worker_loss,axis=0)

        Am = self.r-self.manager_vf
        manager_vf_loss = .5*tf.reduce_sum(tf.square(Am))

        Aw = (self.r + self.config.alpha*self.ri)-self.worker_vf
        worker_vf_loss = .5*tf.reduce_sum(tf.square(Aw))

        entropy = -tf.reduce_sum(self.pi * self.log_pi)

        beta = tf.train.polynomial_decay(self.config.beta_start, self.global_step,
                                         end_learning_rate=self.config.beta_end,
                                         decay_steps=self.config.decay_steps,
                                         power=1)

        # worker_loss = tf.Print(worker_loss,[manager_loss,worker_loss,manager_vf_loss,worker_vf_loss,entropy])
        self.loss = worker_loss+manager_loss+ \
                    worker_vf_loss + manager_vf_loss- \
                    entropy*beta

        bs = tf.to_float(tf.shape(self.obs)[0])
        tf.summary.scalar("model/manager_loss", manager_loss / bs)
        tf.summary.scalar("model/worker_loss", worker_loss / bs)
        tf.summary.scalar("model/value_mean", tf.reduce_mean(self.manager_vf))
        tf.summary.scalar("model/value_loss", manager_vf_loss / bs)
        tf.summary.scalar("model/value_loss_scaled", manager_vf_loss / bs * .5)
        tf.summary.scalar("model/entropy", entropy / bs)
        tf.summary.scalar("model/entropy_loss_scaleed", -entropy / bs * beta)
        # tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        tf.summary.scalar("model/var_global_norm", tf.global_norm(tf.get_collection( \
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)))
        tf.summary.scalar("model/beta", beta)
        tf.summary.image("model/state", self.obs)
        self.summary_op = tf.summary.merge_all()


    def get_initial_features(self):
        return np.zeros((1,1,self.g_dim),np.float32),self.worker_lstm.state_init+self.manager_lstm.state_init


    def act(self, ob, g,cw,hw,cm,hm):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.manager_vf, self.g, self.s, self.last_c_g] + self.state_out,
                        {self.obs: [ob], self.state_in[0]: cw, self.state_in[1]: hw, \
                         self.state_in[2]: cm, self.state_in[3]: hm, \
                         self.prev_g: g})

    def value(self, ob, g, cw, hw, cm, hm):
        sess = tf.get_default_session()
        return sess.run(self.manager_vf,
                        {self.obs: [ob], self.state_in[0]: cw, self.state_in[1]: hw, \
                         self.state_in[2]: cm, self.state_in[3]: hm, \
                         self.prev_g: g})[0]

    def update_batch(self,batch):
        return self.batch_processor.process_batch(batch)
