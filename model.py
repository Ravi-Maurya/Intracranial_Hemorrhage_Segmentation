try:
    import numpy as np
    from cv2 import cv2
    import pydicom
    import keras
    import keras.backend as K
    from keras.layers import Dense, Dropout
    from keras.models import Model,model_from_json
    import efficientnet.keras as efn
except ImportError:
    from pip._internal import main as pip
    pip(['install', '--user', 'numpy==1.16.4','opencv-python==4.2.0.32','pydicom==1.4.2','Keras==2.2.5','efficientnet==1.1.0','h5py==2.9.0'])
    import numpy as np
    from cv2 import cv2
    import pydicom
    import keras
    import keras.backend as K
    from keras.layers import Dense, Dropout
    from keras.models import Model,model_from_json
    import efficientnet.keras as efn

class Transform:
    def __init__(self,path,SHAPE):
        self.path = path
        self.dcm = pydicom.dcmread(self.path)
        self.SHAPE = SHAPE
    
    def correct_dcm(self):
        x = self.dcm.pixel_array + 1000
        px_mode = 4096
        x[x>=px_mode] = x[x>=px_mode] - px_mode
        self.dcm.PixelData = x.tobytes()
        self.dcm.RescaleIntercept = -1000

    def window_image(self, window_center, window_width):    
        if (self.dcm.BitsStored == 12) and (self.dcm.PixelRepresentation == 0) and (int(self.dcm.RescaleIntercept) > -100):
            self.correct_dcm()
        img = self.dcm.pixel_array * self.dcm.RescaleSlope + self.dcm.RescaleIntercept
        img = cv2.resize(img, self.SHAPE[:2], interpolation = cv2.INTER_LINEAR)
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        return img

    def bsb_window(self):
        brain_img = self.window_image(40, 80)
        subdural_img = self.window_image(80, 200)
        soft_img = self.window_image(40, 380)
        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img - (-20)) / 200
        soft_img = (soft_img - (-150)) / 380
        bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)
        return bsb_img

    def read(self):
        try:
            img = self.bsb_window()
        except:
            img = np.zeros(self.SHAPE)
        return img

class DeepModel:
    def __init__(self, BATCH_SIZE=16, HEIGHT=256, WEIGHT=256, CHANNEL=3):
        self.tmp = open('model/model.json','r')
        self.loaded_model = self.tmp.read()
        self.tmp.close()
        self.model = None
        self.BATCH_SIZE = BATCH_SIZE
        self.SHAPE = (HEIGHT, WEIGHT, CHANNEL)
        self.IMAGE_DIR = 'images/'
        self.OUTPUT = ["Any","Epidural","Intraparenchymal","Intraventricular","Subarachnoid","Subdural"]
        self.X = None
        self.Y = None
    
    def create(self):
        self.model = model_from_json(self.loaded_model)
        self.model.load_weights('model/model.h5')
    
    def predict(self,ID):
        self.X = np.empty((self.BATCH_SIZE, *self.SHAPE))
        self.X[0,] = Transform(self.IMAGE_DIR+ID+".dcm", self.SHAPE).read()
        self.Y = np.average(self.model.predict(self.X),axis=0)
    
    def result(self):
        print("The chance of having {} is {}".format(self.OUTPUT[self.Y.argmax()], self.Y.max()))
    
    def probability_table(self):
        for i in range(6):
            print("{}: {}".format(self.OUTPUT[i],self.Y[i]))
    


if __name__ == "__main__":
    ID = "ID_000000e27"
    model = DeepModel(BATCH_SIZE=8)
    model.create()
    model.predict(ID)
    model.result()
    model.probability_table()