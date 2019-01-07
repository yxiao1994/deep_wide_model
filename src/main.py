from src.keras_model import Deep_Wide_Model
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

if __name__ == '__main__':
    origin_train = pd.read_csv('data_after_merge.csv')
    click_features = pd.read_csv('click_features.csv')
    data = pd.concat([origin_train, click_features], axis=1)
    dense_features = ['age', 'user_day_click_num', 'user_hour_click_num', 'user_minute_click_num',
                      'user_day_click_times', 'app_day_hot', 'app_hour_hot']
    sparse_features = [feat for feat in data.columns if feat not in dense_features and feat != 'label']
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_dict = {feat: data[feat].nunique() for feat in sparse_features}
    dense_feature_list = dense_features
    model_input = [data[feat].values for feat in sparse_feature_dict] + [data[feat].values for feat in
                                                                         dense_feature_list]

    model = Deep_Wide_Model(model_input, sparse_feature_dict, dense_feature_list, 8, 0.001, 32)



