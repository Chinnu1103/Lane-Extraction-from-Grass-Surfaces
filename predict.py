from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2

def display(img):
	cv2.namedWindow('frame', 0)
	cv2.imshow('frame', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input_img', help='input test image')
args = parser.parse_args()

model = load_model('lanes.h5')

img = cv2.imread(args.input_img)
img = cv2.copyMakeBorder(img,140,140,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
img = cv2.resize(img, (192, 192))
img = img / 255

mask = model.predict([[img]])
mask = np.resize(mask, (192,192,1))
mask = np.round(mask)
mask = mask * 255
display(mask)