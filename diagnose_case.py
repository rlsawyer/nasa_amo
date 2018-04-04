from sys import argv, exit
from Pipeline.pipeline import InformationRetreival, Preprocess

from os import listdir, getcwd, mkdir, remove
from os.path import isdir, join
from shutil import rmtree
import keras
from keras.applications import VGG16
from PIL import Image
from skimage.transform import resize
import numpy as np

class Diagnose():

    def __init__(self):

        self.Filename = ''
        self.Path = getcwd()
        self.FileExt = '.jpg'
        self.SavePath = join(self.Path, 'processed')
        if self.FileExt == '.mp4':
            self.Mode = 'bmode'
        else:
            self.Mode = 'mmode'

    def set_filename(self,filename):
        self.Filename = filename
        self.FileExt = filename[len(filename)-4:len(filename)]
        if self.FileExt == '.mp4':
            self.Mode = 'bmode'
        else:
            self.Mode = 'mmode'
        self.ModelFilename = 'vgg_mmode_preprocessing_0.0001.h5'

    def set_path(self,path):
        self.Path = path
        self.SavePath = join(self.Path, 'processed')

    def set_model_filename(self, filename):
        self.ModelFilename = filename

    def diagnose_image(self):
        info = InformationRetreival()
        process = Preprocess()

        save_dir = self.SavePath
        if not isdir(save_dir):
            mkdir(save_dir)
        else:
            prev_files = listdir(save_dir)
            for f in prev_files:
                if isdir(f):
                    rmtree(join(save_dir, f))
                else:
                    remove(join(save_dir, f))

        nFrames = 20
        if self.Mode == 'bmode':
            frames_path = join(self.Path, 'frames')
            if not isdir(frames_path):
                mkdir(frames_path)
            info.extract_frames_test_video(self.Filename, self.Path, nFrames)
            img_files = listdir(frames_path)
            for fn in img_files:
                if '.jpg' not in fn:
                    continue
                process.preprocess_test_image(fn, frames_path, self.Mode)
                self.SavePath = join(frames_path, 'processed')
        else:
            process.preprocess_test_image(self.Filename, self.Path, self.Mode)

        model = keras.models.load_model(self.ModelFilename)
        vgg_conv = VGG16(weights='imagenet',
                         include_top=False,
                         input_shape=(224, 224, 3))
        img_files = listdir(self.SavePath)
        pred = []
        prob = []
        for f in img_files:
            if self.Filename[0:len(self.Filename)-4] in f:
                img = Image.open(join(self.SavePath, f))
                img = np.asarray(img, dtype=np.float32).copy()
                img /= 255
                img = resize(img, (224, 224, 3))
                img = np.reshape(img,(1,224,224,3))
                features = vgg_conv.predict(img)

                features = np.reshape(features, (1, 7 * 7 * 512))
                predictions = model.predict_classes(features)
                pred.append(predictions[0])
                probabilities = model.predict(features)
                prob.append(probabilities[0][1])

        return pred, prob


def main():

    filename = ''
    path = getcwd()

    if len(argv) > 1:
        for i in range(len(argv)):
            if argv[i] == '--f':
                nm = argv[i+1]
                if '/' in nm:
                    spl = nm.split('/')
                    filename = spl[len(spl)-1]
                    idx = nm.find(filename)
                    path = nm[0:idx-1]
                else:
                    filename = nm
            if argv[i] == '--p':
                path = argv[i+1]

    if filename == '':
        if isdir(path):
            fls = listdir(path)
            for fl in fls:
                diag = Diagnose()
                diag.set_filename(fl)
                diag.set_path(path)
                pred = diag.diagnose_image()
        else:
            exit('Error: No file or folder specified.')
    else:
        diag = Diagnose()
        diag.set_filename(filename)
        diag.set_path(path)
        pred, prob = diag.diagnose_image()

        print('Prediction is', pred)
        print('Probability of positive is', prob)
        print('Mean prediction is', np.mean(pred))
        print('Mean probability of positive is', np.mean(prob))


if __name__ == '__main__':
    main()

