import numpy as np
import pydicom
import keras
import keras.backend as K
from keras.layers import Dense, Dropout
from keras.models import Model,model_from_json
import efficientnet.keras as efn

SHAPE = (256,256,3)
IMAGE_DIR = 'images/'
BATCH_SIZE = 16
OUTPUT = ["Any","Epidural","Intraparenchymal","Intraventricular","Subarachnoid","Subdural"]

tmp = open('model/model.json','r')
loaded_model = tmp.read()
tmp.close()
model = model_from_json(loaded_model)
model.load_weights('model/model.h5')
def _read(path,dimension):
    dcm = pydicom.dcmread(path)
    img = np.zeros(dimension)
    return img
X = np.empty((BATCH_SIZE, *SHAPE))
ID = "ID_000000e27"
image = _read(IMAGE_DIR+ID+".dcm",SHAPE)
X[0,] = image
Y = np.nanmean(model.predict(X),axis=0)
print(Y[Y.argmax()],OUTPUT[Y.argmax()])
