from src.data_parser import DataParser
from src.model import deep_wide_model
import argparse
import pandas as pd

if __name__ == '__main__':
    df_train = pd.read_csv('df_train.csv')
    dp = DataParser()
    dfi_train, dfv_train, dfi_test, dfv_test, feature_dim, target = dp.generate_data()
