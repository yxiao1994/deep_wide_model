import tensorflow as tf
import numpy as np


class deep_wide_model(object):
    def __init__(self, feature_size, field_size, embedding_dim, dropout_rate, batch_size, loss_type='logloss',
                 deep_layers=[64, 32], activation=tf.nn.relu, epoch=5):
        self.feature_size = feature_size  # denoted as N, feature size after one-hot encoding
        self.field_size = field_size  # denoted as F, feature size before one-hot encoding
        self.embedding_size = embedding_dim  # denoted as K, embedding size
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.deep_layers = deep_layers
        self.activation = activation
        self.epoch = epoch
        self.train_result, self.valid_result = [], []

    def _init_weights(self):
        weights = dict()
        # wide部分权重，N * 1
        weights['wide'] = tf.Variable(tf.truncated_normal([self.feature_size, 1], 0, 0.01),
                                      name='lr weights')
        # deep部分权重，N * K
        weights['deep'] = tf.Variable(tf.truncated_normal([self.feature_size, 1], 0, 0.01),
                                      name='embedding weights')

        # 全连接层的权重
        num_layers = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights['deep_layer_weights_0'] = tf.Variable(
            tf.random_normal([input_size, self.deep_layers[0]], 0, glorot))
        weights['deep_layer_biase_0'] = tf.Variable(
            tf.random_normal([1, self.deep_layers[0]], 0, glorot))

        for i in range(1, num_layers):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights['deep_layer_weights_%d' % i] = tf.Variable(
                tf.random_normal([self.deep_layers[i - 1], self.deep_layers[0]], 0, glorot))
            weights['deep_layer_biase_%d' % i] = tf.Variable(
                tf.random_normal([1, self.deep_layers[i]], 0, glorot))

        # 最后的输出层权重
        input_size = self.field_size + self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(
            tf.random_normal([input_size, 1], 0, glorot))
        weights['concat_biase'] = tf.Variable(
            tf.random_normal(tf.constant(0.01)), dtype=tf.float32)

        return weights

    def _init_graph(self):
        with tf.Graph.as_default():
            self.sess = tf.InteractiveSession()
            self.feature_index = tf.placeholder(tf.int32, [None, self.field_size], name='feature index')
            self.feature_value = tf.placeholder(tf.int32, [None, self.field_size], name='feature value')
            self.weights = self._init_weights()
            self.y = tf.placeholder(tf.int32, [None, 1], name='label')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.lr = tf.nn.embedding_lookup(self.weights['wide'], self.feature_index)  # None * F * 1
            self.lr = tf.squeeze(self.lr)  # None * F
            self.lr = self.lr * self.feature_value  # None * F

            self.deep = tf.nn.embedding_lookup(self.weights['deep'], self.feature_index)  # None * F * K
            self.deep = tf.reshape(self.deep, [-1, self.embedding_size * self.field_size])  # None * (F * K)
            for i in range(len(self.deep_layers)):
                self.deep = self.deep * self.weights['deep_layer_weights_%d' % i]
                self.deep = self.deep + self.weights['deep_layer_biase_%d' % i]
                self.deep = self.activation(self.deep)
                self.deep = tf.nn.dropout(self.deep, self.dropout_rate)

            self.output = tf.concat([self.deep, self.lr], axis=1)
            self.output = tf.add(tf.multiply(self.output, self.weights['concat_projection']),
                                 self.weights['concat_biase'])

            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.y, self.out)
            if self.loss_type == 'l2_loss':
                self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.out))

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            tf.global_variables_initializer().run()

    def get_batch_data(self, Xi, Xv, label, index):
        if len(label.shape) == 1:
            label = np.reshape(label, [label.shape[0], 1])
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        end = end if end < len(label) else len(label)
        return Xi[start:end], Xv[start:end], label[start, end]

    def shuffle_data(self, Xi, Xv, label):
        data_size = len(label)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        return Xi[shuffle_indices], Xv[shuffle_indices], label[shuffle_indices]

    def fit_batch_data(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.train_phase: True}
        loss, _ = self.sess.run([self.loss, self.optimizer])
        return loss

    def fit(self, Xi_train, Xv_tain, y_train, Xi_valid=None, Xv_valid=None, y_valid=None):
        has_valid = (Xi_valid is not None)
        total_batch = int(len(y_train) / self.batch_size)
        for epoch in self.epoch:
            Xi_train, Xv_tain, y_train = self.shuffle_data(Xi_train, Xv_tain, y_train)
            for batch_index in total_batch:
                Xi_batch, Xv_batch, y_batch = self.get_batch_data(Xi_train, Xv_tain, y_train)
                self.fit_batch_data(Xi_batch, Xv_batch, y_batch)

            train_result = self.evaluate(Xi_train, Xv_tain, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result))
                else:
                    print("[%d] train-result=%.4f "
                          % (epoch + 1, train_result))

    def predict(self, Xi_test, Xv_test):
        total_batch = len(Xi_test) / self.batch_size
        y_predict = np.array([])
        y_dummy = [1] * len(Xi_test)
        for batch_index in range(total_batch + 1):
            Xi_batch, Xv_batch, y_batch = self.get_batch_data(Xi_test, Xv_test, y_dummy, batch_index)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.y: y_batch,
                         self.train_phase: False
                         }
            batch_res = self.sess.run(self.out, feed_dict=feed_dict)
            y_predict = np.concatenate([y_predict, batch_res.reshape(-1,)])

        return y_predict

    def evaluate(self, Xi, Xv, y_true):
        y_predict = self.predict(Xi, Xv)
        return tf.losses.log_loss(y_true, y_predict)
