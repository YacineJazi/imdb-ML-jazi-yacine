import argparse
import json
import os
import sys
import traceback
from glob import glob
import math
import random
import numpy as np
from dotenv import load_dotenv
from azureml.core import Dataset, Datastore, Experiment, Run, Workspace
from azureml.core.authentication import AzureCliAuthentication
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dotnetcore2 import runtime
runtime.version = ("18", "10", "0")

load_dotenv()


def main():
    cli_auth = AzureCliAuthentication()

    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    train_test_data_folder = os.path.join(os.environ.get('ROOT_DIR'), 'data/tmp/train_test_data')
    os.makedirs(train_test_data_folder, exist_ok=True)


    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=None
    )
    datastore = Datastore(ws)
    dataset = Dataset.get_by_name(ws, name='ratings')
    dataset = dataset.to_pandas_dataframe()
    dataset= formatData(dataset)
    dataset_train,dataset_test = splitData(dataset)
    saveNumpyArrays(train_test_data_folder, dataset_train=dataset_train, dataset_test=dataset_test)
    datastore.upload(src_dir=train_test_data_folder, target_path='imdb_train_test')
    train_test_data = Dataset.File.from_files([(datastore, 'imdb_train_test')],validate=False)
    train_test_data.register(
    workspace=ws,
    name="imdb_train_test",
    description="Processed train- and testdata for consumption by model",
    create_new_version=True
    )

def formatData(dataset):
    dataset= dataset.rename({"Column1": "reviewer", "Column2": "movie", "Column3": "score"},axis="columns")
    dataset= dataset.pivot(index="reviewer",columns="movie",values="score").fillna(0).astype(int)
    i=1
    for index in dataset.columns:
        while(index!=i):
            dataset.insert(i,i,[0]*6040)
            i+=1
        i+=1
    dataset.add_prefix('movie')
    return dataset

def splitData(dataset):
    dataset_test = dataset 
    dataset_train = dataset
    for index, row in dataset_train.iterrows():
        non_null=row.to_numpy().nonzero()[0]
        indexes = random.sample(range(0,len(row.to_numpy().nonzero()[0])-1), 10)
        for i in indexes:
            dataset_train.iat[index-1,non_null[i]]=0
    return dataset_train.div(5),dataset_test.div(5)

def saveNumpyArrays(folder, **arrays):
    for array_name, array in arrays.items():
        np.save(f"{folder}/{array_name}.npy", array)

if __name__ == '__main__':
    main()