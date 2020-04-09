try:
    import numpy as np
    from cv2 import cv2
    import pydicom
    import keras
    from keras.layers import Dense, Dropout
    from keras.models import Model, model_from_json
    from keras import backend as K
    import efficientnet.keras as efn
    import plotly
    import plotly.graph_objs as go
    import plotly.io as pio
    import json
    import gc
except ImportError:
    from pip._internal import main as pip

    pip(['install', '--user', 'numpy==1.16.4', 'opencv-python==4.2.0.32', 'pydicom==1.4.2', 'Keras==2.2.5',
         'efficientnet==1.1.0', 'h5py==2.9.0'])
    import numpy as np
    from cv2 import cv2
    import pydicom
    import keras
    from keras.layers import Dense, Dropout
    from keras.models import Model, model_from_json
    import efficientnet.keras as efn
    import plotly.graph_objs as go
    from matplotlib.pyplot import plot

class Transform:
    def __init__(self, path, shape):
        self.path = path
        self.dcm = pydicom.dcmread(self.path)
        self.shape = shape

    def correct_dcm(self):
        x = self.dcm.pixel_array + 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode
        self.dcm.PixelData = x.tobytes()
        self.dcm.RescaleIntercept = -1000

    def window_image(self, window_center, window_width):
        if (self.dcm.BitsStored == 12) and (self.dcm.PixelRepresentation == 0) and (
                int(self.dcm.RescaleIntercept) > -100):
            self.correct_dcm()
        img = self.dcm.pixel_array * self.dcm.RescaleSlope + self.dcm.RescaleIntercept
        img = cv2.resize(img, self.shape[:2], interpolation=cv2.INTER_LINEAR)
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
        bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)
        return bsb_img

    def read(self):
        try:
            img = self.bsb_window()
        except:
            img = np.zeros(self.shape)
        return img


class DeepModel:
    def __init__(self, batch_size=16, height=256, width=256, channel=3):
        self.tmp = open('model/model.json', 'r')
        self.loaded_model = self.tmp.read()
        self.tmp.close()
        self.model = None
        self.batch_size = batch_size
        self.shape = (height, width, channel)
        self.IMAGE_DIR = 'images/'
        self.OUTPUT = ["Any", "Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]
        self.X = None
        self.Y = None

    def create(self):
        self.model = model_from_json(self.loaded_model)
        self.model.load_weights('model/model.h5')

    def destroy(self):
        del self.model
        gc.collect()
        self.model = None
        K.clear_session()

    def predict(self, id_name):
        self.Y = None
        self.X = None
        self.id_name = id_name
        self.X = np.empty((self.batch_size, *self.shape))
        self.X[0, ] = Transform(self.IMAGE_DIR + self.id_name, self.shape).read()
        self.Y = np.average(self.model.predict(self.X), axis=0)

    def result(self):
        return self.OUTPUT[self.Y.argmax()]

    def probability_table(self):
        for i in range(6):
            print("{}: {}".format(self.OUTPUT[i], self.Y[i]))

    def show_plot(self):
        fig = go.Figure([go.Bar(x = self.OUTPUT, y = self.Y)])
        fig.write_html("templates/fig.html")
        return 'home/ravi/Desktop/GitHub/Intracranial_Hemorrhage_Segmentation/templates/fig.html'



# if __name__=='__main__':
#     model = DeepModel()
#     model.create()
#     model.predict('ID_000000e27')
#     model.show_plot()