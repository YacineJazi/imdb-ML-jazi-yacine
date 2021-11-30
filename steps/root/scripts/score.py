import json
import numpy as np
import os
from tensorflow.keras.models import load_model

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join("./azureml-models", 'imdb-model')
    model = load_model(model_path)

def run(data):
    data = json.loads(data)
    result = model.predict(np.asarray(data))
    return {"result": result.tolist()}
