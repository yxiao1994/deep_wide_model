# coding:utf-8
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.initializers import RandomNormal
from keras.regularizers import l2
from src.layers import FM, CrossNet


class Deep_Wide_Model(object):
    def __init__(self, train_data, sparse_feature_dim_dict, dense_feature_list, embedding_size, learning_rate,
                 batch_size):
        """

        :param sparse_feature_dim_dict: records the field size of sparse features
        :param dense_feature_list: list of dense features
        :param embedding_size: embedding size
        :param learning_rate: learning rate
        :param batch_size: batch size
        """
        self.train_data = train_data
        self.sparse_feature_dim_dict = sparse_feature_dim_dict
        self.dense_feature_list = dense_feature_list
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _get_input(self):
        sparse_input = [Input(shape=(1,), dtype='int32', name='sparse_input_' + feat)
                        for feat in self.sparse_feature_dim_dict]
        dense_input = [Input(shape=(1,), dtype='float32', name='dense_input_' + feat)
                       for feat in self.dense_feature_list]
        return sparse_input, dense_input

    def _get_embedding(self):
        sparse_embedding = [Embedding(self.sparse_feature_dim_dict[feature], self.embedding_size,
                                      embeddings_initializer=RandomNormal(
                                          mean=0.0, stddev=0.0001),
                                      embeddings_regularizer=l2(0.00001),
                                      ) for feature
                            in self.sparse_feature_dim_dict]
        linear_embedding = [Embedding(self.sparse_feature_dim_dict[feature], 1,
                                      embeddings_initializer=RandomNormal(mean=0.0, stddev=0.0001),
                                      embeddings_regularizer=l2(0.00001), ) for feature
                            in self.sparse_feature_dim_dict]
        return sparse_embedding, linear_embedding

    def deep_wide_model(self):
        sparse_input, dense_input = self._get_input()
        sparse_embedding, linear_embedding = self._get_embedding()

        embed_list = [sparse_embedding[i](sparse_input[i])
                      for i in range(len(sparse_input))]
        deep_part = Concatenate(axis=1)(embed_list)  # None * F * K
        deep_part = Flatten()(deep_part)  # None * (F * K)

        if len(dense_input) > 0:
            deep_part = Concatenate()([deep_part] + dense_input)
        deep_part = Dense(32, activation='relu')(deep_part)  # None * 32
        deep_part = Dense(16, activation='relu')(deep_part)  # None * 16

        bias_embed_list = [linear_embedding[i](sparse_input[i])
                           for i in range(len(self.sparse_feature_dim_dict))]
        wide_part = Flatten()(Add()(bias_embed_list))  # None * 1

        print(deep_part.shape)
        print(wide_part.shape)

        merged = Concatenate()([wide_part, deep_part])

        output = Dense(1, activation="sigmoid")(merged)
        model = Model(inputs=sparse_input + dense_input, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def deep_fm_model(self):
        sparse_input, dense_input = self._get_input()
        sparse_embedding, linear_embedding = self._get_embedding()

        embed_list = [sparse_embedding[i](sparse_input[i])
                      for i in range(len(sparse_input))]

        bias_embed_list = [linear_embedding[i](sparse_input[i])
                           for i in range(len(self.sparse_feature_dim_dict))]
        wide_part = Flatten()(Add()(bias_embed_list))  # None * 1

        if len(dense_input) > 0:
            continuous_embedding_list = [Dense(self.embedding_size)(x) for x in dense_input]
            continuous_embedding_list = [Reshape((1, self.embedding_size))(x) for x in continuous_embedding_list]
            embed_list += continuous_embedding_list
            dense_part = dense_input[0] if len(dense_input) == 1 else Concatenate()(dense_input)
            dense_part = Dense(1, activation=None, use_bias=False)(dense_part)
            wide_part = Add()([wide_part, dense_part])  # None * 1

        fm_input = Concatenate(axis=1)(embed_list)  # None * F * K
        fm_part = FM()(fm_input)

        deep_part = Flatten()(fm_input)  # None * (F * K)
        deep_part = Dense(128, activation='relu')(deep_part)  # None * 32
        deep_part = Dense(128, activation='relu')(deep_part)  # None * 32
        deep_part = Dense(1)(deep_part)  # None * 1

        merged = Concatenate()([deep_part, wide_part, fm_part])
        output = Dense(1, activation="sigmoid")(merged)
        model = Model(inputs=sparse_input + dense_input, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'], )
        return model

    def dcn_model(self):
        sparse_input, dense_input = self._get_input()
        sparse_embedding, linear_embedding = self._get_embedding()

        embed_list = [sparse_embedding[i](sparse_input[i])
                      for i in range(len(sparse_input))]
        deep_input = Flatten()(Concatenate()(embed_list))  # None * (F * K)
        if len(dense_input) > 0:
            if len(dense_input) == 1:
                continuous_list = dense_input[0]
            else:
                continuous_list = Concatenate()(dense_input)

            deep_input = Concatenate()([deep_input, continuous_list])

        cross_part = CrossNet()(deep_input)
        deep_part = Dense(128, activation='relu')(deep_input)
        deep_part = Dense(32, activation='relu')(deep_input)
        merged = Concatenate()([cross_part, deep_part])
        output = Dense(1, activation="sigmoid")(merged)
        model = Model(inputs=sparse_input + dense_input, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'], )
        return model

    def fit(self, model_name):
        if model_name == 'deep_wide_model':
            model = self.deep_wide_model()
        elif model_name == 'deep_fm_model':
            model = self.deep_fm_model()
        elif model_name == 'dcn_model':
            model = self.dcn_model()
        else:
            print('please choose the right model! ')
        print(model.summary())
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)
        best_model_path = "deep_wide_model" + ".h5"
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
        input1 = [self.train_data[feat].values for feat in self.sparse_feature_dim_dict]
        input2 = [self.train_data[feat].values for feat in self.dense_feature_list]

        hist = model.fit(input1 + input2, self.train_data['label'].values, validation_split=0.2,
                         epochs=15, batch_size=self.batch_size, shuffle=True,
                         callbacks=[early_stopping, model_checkpoint],
                         verbose=1)
        model.load_weights(best_model_path)
        model.save('deep-wide-model.h5')
