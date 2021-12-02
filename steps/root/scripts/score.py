import json
import numpy as np
import os
from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow.keras import backend as K

def custom_loss(y_true,y_pred):
    y_mask=tf.keras.backend.clip(y_true, 0, 0.01)*100
    return K.mean(K.square(y_mask*(y_pred - y_true)), axis=-1)

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'imdb_model')
    model = load_model(model_path,custom_objects={'custom_loss': custom_loss})

#!!Preprocess here!!
def run(data):
    try:
        print(model)
        data = json.loads(data)
        data = np.asarray(data["data"])
        result = model.predict(data)
        return {"result": result.tolist()}
    except Exception as e:
        result = str(e)
        return {"error": e}

#