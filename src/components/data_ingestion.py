import os
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
from data_transformation import DataTransformation


@dataclass
class DataIngestionconfig:
    train_path=os.path.join('artifact','train.csv')
    test_path=os.path.join('artifact','test.csv')
    raw_path=os.path.join('artifact','raw.csv')

class DataIngestion:
    def __init__(self):
        self.x=DataIngestionconfig()
    def intiate_data_ingestion(self):
        datasetPath="Notebook/dataset/Tiroid.csv" 
        df=pd.read_csv(datasetPath)

        os.makedirs(os.path.dirname(self.x.train_path),exist_ok=True)
        df.to_csv(self.x.raw_path,index=False)

        train_set,test_set=train_test_split(df,test_size=0.2,random_state=1000)

        df.to_csv(self.x.test_path,index=False)
        df.to_csv(self.x.train_path,index=False)

        return self.x.test_path,self.x.raw_path,self.x.train_path
    

if __name__=='__main__':
        obj=DataIngestion()
        train_set,test_set,_=obj.intiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.insert_trainingAnd_test(train_set,test_set)

