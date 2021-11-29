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
import tensorflow.keras
from tensorflow.keras.layers import Flatten, Input, concatenate, Dense, Activation, Dropout, BatchNormalization,  MaxPooling2D, AveragePooling2D, Conv2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from azureml.core import Run
from dotnetcore2 import runtime
from azureml.core import Experiment,ScriptRunConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os


def main():
    cli_auth = AzureCliAuthentication()
    #Get env variables
    workspace_name = os.environ.get("WORKSPACE_NAME")
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    env_name = os.environ.get("AML_ENV_NAME")
    model_name = os.environ.get("MODEL_NAME")
    dataset_name = os.environ.get("TRAINING_TESTING_DATASET")
    script_folder = os.path.join(os.environ.get('ROOT_DIR'), 'scripts')
    config_state_folder = os.path.join(os.environ.get('ROOT_DIR'), 'config_states')

    #Create folder if not exist
    train_test_data_folder = os.path.join(os.environ.get('ROOT_DIR'), 'data/tmp/train_test_data')
    os.makedirs(train_test_data_folder, exist_ok=True)

    #Connect to workspace
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )

    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    compute_target = prepareMachines(ws)
    env = prepareEnv(ws, env_name)
    src = prepareTraining(dataset, model_name, script_folder, compute_target, env)
    exp = Experiment(workspace=ws, name=experiment_name)
    run = exp.submit(config=src)
    run.wait_for_completion()
    run_details = {k:v for k,v in run.get_details().items() if k not in ['inputDatasets', 'outputDatasets']}
    
    if not os.path.exists(os.path.dirname(config_state_folder)):
        try:
            os.makedirs(os.path.dirname(config_state_folder))
        except OSError as exc: # Guard against race condition
            raise


    with open('filename', 'w') as training_run_json:
        json.dump(run_details, training_run_json)
    
    


def prepareMachines(ws):
    #Set variables
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "imdb-cluster")
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)
    # This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_NC6")
    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print("found compute target: " + compute_name)
    else:
        print("creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                    min_nodes = compute_min_nodes, 
                                                                    max_nodes = compute_max_nodes,
                                                                    identity_type="SystemAssigned")
        # create the cluster
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    return compute_target

def prepareTraining(dataset, model_name, script_folder, compute_target, env):
    args = ['--data-folder', dataset.as_mount(), '--epochs', 20, '--batch_size', 5, '--model_name', model_name]
    src = ScriptRunConfig(source_directory=script_folder,
                        script='train.py', 
                        arguments=args,
                        compute_target=compute_target,
                        environment=env)
    return src

def prepareEnv(ws, env_name):
    env = Environment(env_name)
    cd = CondaDependencies.create(
        pip_packages=['azureml-dataset-runtime[pandas,fuse]', 'azureml-defaults', 'tensorflow', 'scikit-learn','pandas'],
        )
    env.python.conda_dependencies = cd
    # Register environment to re-use later
    env.register(workspace = ws)
    return env
