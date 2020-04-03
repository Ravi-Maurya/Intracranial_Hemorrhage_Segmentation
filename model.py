import numpy as np
from cv2 import cv2
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
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    
    # Resize
    img = cv2.resize(img, SHAPE[:2], interpolation = cv2.INTER_LINEAR)
   
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)
    return bsb_img
def _read(path,dimension):
    dcm = pydicom.dcmread(path)
    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(dcm)
    return img

tmp = open('model/model.json','r')
loaded_model = tmp.read()
tmp.close()
model = model_from_json(loaded_model)
model.load_weights('model/model.h5')
X = np.empty((BATCH_SIZE, *SHAPE))
ID = "ID_000000e27"
image = _read(IMAGE_DIR+ID+".dcm",SHAPE)
X[0,] = image
Y = np.nanmean(model.predict(X),axis=0)
print(Y[Y.argmax()],OUTPUT[Y.argmax()])
