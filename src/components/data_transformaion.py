import os
import sys
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import sparse

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from src.utils import save_object

sys.setrecursionlimit(5000)

@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
            'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
            'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
            'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'MiscVal', 'MoSold', 'YrSold']
            cat_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
            'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
            'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
            'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
            'SaleType', 'SaleCondition']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler(with_mean=False)),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown='infrequent_if_exist')),
                    ('scaler', StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipepine", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns),
            ])

            return preprocessor
        
        except Exception as e:
            raise e
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj=self.get_data_transformer_obj()

            target_column = 'SalePrice'

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            print("Shape",input_feature_train_arr.shape, target_feature_train_df.shape)

            train_arr = np.c_[input_feature_train_arr.toarray(),np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr.toarray(), np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise e