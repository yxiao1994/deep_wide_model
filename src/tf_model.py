import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


class deep_wide_model(object):
    def __init__(self, feature_size, field_size, embedding_dim, batch_size, loss_type='logloss',
                 deep_layers=[64, 32], activation=tf.nn.relu, use_fm=True, epoch=5,
                 path='/home/shenyu/notebooks/xy/tencent_ad/models/', learning_rate=0.001, verbose=2):
        self.use_fm = use_fm
        self.feature_size = feature_size  # denoted as N, feature size after one-hot encoding
        self.field_size = field_size  # denoted as F, feature size before one-hot encoding
        self.embedding_size = embedding_dim  # denoted as K, embedding size
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.deep_layers = deep_layers
        self.activation = activation
        self.epoch = epoch
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.train_result, self.valid_result = [], []
        self.path = path
        self._init_graph()

    def _init_weights(self):
        weights = dict()
        # wide部分权重，N * 1
        weights['feature_bias'] = tf.Variable(tf.truncated_normal([self.feature_size, 1], mean=0, stddev=0.0001),
                                              name='lr_weights')
        # deep部分权重，N * K
        weights['feature_embeddings'] = tf.Variable(
            tf.truncated_normal([self.feature_size, self.embedding_size], mean=0, stddev=0.0001),
            name='embedding_weights')

        # 全连接层的权重
        num_layers = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights['deep_layer_weights_0'] = tf.Variable(
            tf.random_normal([input_size, self.deep_layers[0]], 0, glorot), name='deep_layer_weights_0')
        weights['deep_layer_bias_0'] = tf.Variable(
            tf.random_normal([1, self.deep_layers[0]], 0, glorot), name='deep_layer_bias_0')

        for i in range(1, num_layers):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights['deep_layer_weights_%d' % i] = tf.Variable(
                tf.random_normal([self.deep_layers[i - 1], self.deep_layers[i]], 0, glorot),
                name='deep_layer_weights_{}'.format(i))
            weights['deep_layer_bias_%d' % i] = tf.Variable(
                tf.random_normal([1, self.deep_layers[i]], 0, glorot), name='deep_layer_bias_{}'.format(i))

        # 最后的输出层权重
        if self.use_fm:
            input_size = self.field_size + self.deep_layers[-1] + self.embedding_size
            self.model_name = 'deep_fm_model'
        else:
            input_size = self.field_size + self.deep_layers[-1]
            self.model_name = 'deep_wide_model'
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(
            tf.random_normal([input_size, 1], 0, glorot))
        weights['concat_bias'] = tf.Variable(
            tf.constant(0.01), dtype=tf.float32)

        return weights

    def _init_graph(self):
        g = tf.Graph()
        with g.as_default():
            self.sess = tf.InteractiveSession()
            self.feature_index = tf.placeholder(tf.int32, [None, self.field_size], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, [None, self.field_size], name='feature_value')
            self.weights = self._init_weights()
            self.y = tf.placeholder(tf.float32, [None, 1], name='label')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # -------------LR part -------------
            self.lr = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feature_index)  # None * F * 1
            self.lr = tf.reduce_sum(self.lr, axis=2)  # None * F
            self.lr = self.lr * self.feature_value  # None * F
            self.lr = tf.nn.dropout(self.lr, keep_prob=self.keep_prob)

            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],
                                                     self.feature_index)  # None * F * K
            self.feat_value = self.feature_value[:, :, None]  # None * F * 1
            self.fm_input = self.embeddings * self.feat_value  # None * F * K

            # -------------fm part-------------
            self.sum_square_part = tf.square(tf.reduce_sum(self.fm_input, axis=1))  # None * K
            self.square_sum_part = tf.reduce_sum(tf.square(self.fm_input), axis=1)  # None * K
            self.fm = 0.5 * (self.sum_square_part - self.square_sum_part)  # None * K
            self.fm = tf.nn.dropout(self.fm, keep_prob=self.keep_prob)

            # -------------deep part-------------
            self.deep = tf.reshape(self.fm_input, [-1, self.embedding_size * self.field_size])  # None * (F * K)
            for i in range(len(self.deep_layers)):
                self.deep = tf.add(tf.matmul(self.deep, self.weights['deep_layer_weights_%d' % i]),
                                   self.weights['deep_layer_bias_%d' % i])
                self.deep = self.activation(self.deep)
                self.deep = tf.nn.dropout(self.deep, self.keep_prob)

            if self.use_fm:
                self.out = tf.concat([self.deep, self.lr, self.fm], axis=1)
            else:
                self.out = tf.concat([self.deep, self.lr], axis=1)
            self.out = tf.add(tf.matmul(self.out, self.weights['concat_projection']),
                              self.weights['concat_bias'])

            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.y, self.out)
            if self.loss_type == 'l2_loss':
                self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.out))
            tf.summary.scalar('loss', self.loss)
            # summaries合并
            self.merged = tf.summary.merge_all()
            log_dir = '/home/shenyu/notebooks/xy/tencent_ad/log'
            # 写到指定的磁盘路径中
            self.train_writer = tf.summary.FileWriter(log_dir + '/train', self.sess.graph)

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            tf.global_variables_initializer().run()
            self.model_saver = tf.train.Saver()

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch_data(self, Xi, Xv, label, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        end = end if end < len(label) else len(label)
        return Xi[start:end], Xv[start:end], label[start:end]

    def shuffle_data(self, Xi, Xv, label):
        data_size = len(label)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        return Xi[shuffle_indices], Xv[shuffle_indices], label[shuffle_indices]

    def fit_batch_data(self, Xi, Xv, y):
        feed_dict = {self.feature_index: Xi,
                     self.feature_value: Xv,
                     self.y: y,
                     self.train_phase: True,
                     self.keep_prob: 0.5}
        loss, summary, _ = self.sess.run([self.loss, self.merged, self.optimizer], feed_dict=feed_dict)
        return loss, summary

    def fit(self, Xi_train, Xv_tain, y_train, Xi_valid=None, Xv_valid=None, y_valid=None):
        has_valid = (Xi_valid is not None)
        total_batch = int(len(y_train) / self.batch_size)
        train_loss = 0
        best_loss = 1.0
        for epoch in range(self.epoch):
            print('epoch {}'.format(epoch))
            Xi_train, Xv_tain, y_train = self.shuffle_data(Xi_train, Xv_tain, y_train)
            for batch_index in range(total_batch):
                # print('batch index: '.format(batch_index))
                Xi_batch, Xv_batch, y_batch = self.get_batch_data(Xi_train, Xv_tain, y_train, batch_index)
                loss, summary = self.fit_batch_data(Xi_batch, Xv_batch, y_batch)
                train_loss += loss
                if (batch_index + 1) % 100 == 0:
                    print("batch [%d] train-result=%.4f"
                          % (batch_index + 1, train_loss / 100))
                    train_loss = 0
                    self.train_writer.add_summary(summary, batch_index)
            print('analysis train loss...')
            train_result = self.evaluate(Xi_train, Xv_tain, y_train)
            self.train_result.append(train_result)

            if has_valid:
                print('analysis valid loss...')
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                if valid_result < best_loss:
                    self.model_saver.save(self.sess, self.path + self.model_name)
                else:
                    self.model_saver.restore(self.sess, self.path + self.model_name)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f"
                          % (epoch + 1, train_result, valid_result))
                else:
                    print("[%d] train-result=%.4f "
                          % (epoch + 1, train_result))
        self.model_saver.restore(self.sess, self.path + self.model_name)

    def predict(self, Xi_test, Xv_test):
        total_batch = int(len(Xi_test) / self.batch_size)
        y_predict = np.array([])
        y_dummy = np.zeros((len(Xi_test), 1))
        for batch_index in range(total_batch + 1):
            Xi_batch, Xv_batch, y_batch = self.get_batch_data(Xi_test, Xv_test, y_dummy, batch_index)
            feed_dict = {self.feature_index: Xi_batch,
                         self.feature_value: Xv_batch,
                         self.y: y_batch,
                         self.train_phase: False,
                         self.keep_prob: 1.0
                         }
            batch_res = self.sess.run(self.out, feed_dict=feed_dict)
            y_predict = np.concatenate([y_predict, batch_res.reshape(-1, )])
            # print('predict shape: {}'.format(y_predict.shape))

        return y_predict

    def evaluate(self, Xi, Xv, y_true):
        y_true = y_true.reshape((-1,))
        y_predict = self.predict(Xi, Xv)
        return log_loss(y_true, y_predict)
