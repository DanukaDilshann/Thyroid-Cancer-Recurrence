import os
import sys
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path: str = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.x=DataTransformationconfig()
 
    
    def get_dataTransformation(self):
        cat=['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy','Thyroid Function', 'Physical Examination',
              'Adenopathy', 'Pathology','Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']
        con=['Age']

        numpipeline=Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ]
        )

        catpipeline=Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ]
        )

        preprocess=ColumnTransformer(
            transformers=[
                ("num_pipeline", numpipeline, con),
                ("cat_pipeline", catpipeline, cat)                
            ]
        )

        return preprocess
    
    def insert_trainingAnd_test(self,train_path,test_path,target_column_name="MEDV"):
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)

        preprosess_obj=self.get_dataTransformation()
        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]

        input_feature_train_arr = preprosess_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprosess_obj.transform(input_feature_test_df)

        train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values]
        test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values] 


        
        file_path = self.x.preprocessor_obj_file_path  # Correct assignment without comma

        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists

        with open(file_path, 'wb') as file:
            pickle.dump(preprosess_obj, file)  # Save the preprocessor object  


        return train_arr, test_arr,self.x.preprocessor_obj_file_path