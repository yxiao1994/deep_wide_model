import pandas as pd


class DataParser(object):
    def __init__(self, df_train, df_test, numeric_columns=[], ignore_columns=[], target_name='label'):
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.target = self.df_train.pop(target_name).values
        self.numeric_columns = numeric_columns
        self.ignore_columns = ignore_columns
        self._gen_feature_index()

    def _gen_feature_index(self):
        df_train = self.df_train
        df_test = self.df_test

        # drop ignore columns
        df_train.drop(columns=self.ignore_columns, inplace=True)
        df_test.drop(columns=self.ignore_columns, inplace=True)
        df = pd.concat([df_train, df_test])

        # map features to index, one-hot for categorical features
        self.feature_index = {}
        index = 0
        for col in df.columns:
            if col in self.numeric_columns:
                self.feature_index[col] = index
                index += 1
            else:
                feature_value = df[col].unique()
                self.feature_index[col] = dict(zip(feature_value, range(index, index + len(feature_value))))
                index += len(feature_value)

        # feature dim after one-hot
        self.feature_dim = index

        self.df_train = df_train
        self.df_test = df_test

    def _parser_data(self, data):
        dfi = data.copy()
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.numeric_columns:
                dfi[col] = self.feature_index[col]
            else:
                dic = self.feature_index[col]
                dfi[col] = dfi[col].map(dic)
                dfv[col] = 1

        dfi = dfi.values.tolist()
        dfv = dfv.values.tolist()
        return dfi, dfv

    def generate_data(self):
        dfi_train, dfv_train = self._parser_data(self.df_train)
        dfi_test, dfv_test = self._parser_data(self.df_test)
        return dfi_train, dfv_train, dfi_test, dfv_test, self.feature_dim, self.target