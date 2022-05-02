from keras.models import load_model
import numpy as np
import cv2
import os
from keras.applications import resnet
from keras.applications import vgg16
from keras.applications import xception
import time
import os
from upload import upload_blob

# set GPU to be invisible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

img_path = input('Enter image path: ')

img_shape = (224, 224)
labels = ['G. Bulloides', 'G. Ruber', 'G. Sacculifer', 'N. Dutertrei', 'N. Incompta', 'N. Pachyderma', 'Others']

model = load_model("./models/foram.h5")

xception_model = xception.Xception(include_top=False, pooling='avg')
resnet50_model = resnet.ResNet50(include_top=False, pooling='avg')

img_filenames = [img_path + p for p in os.listdir(img_path)]

group_images = np.zeros(img_shape + (len(img_filenames),))
for i, img_file in enumerate(img_filenames):
    img = cv2.imread(img_file, 0)
    saveimg = cv2.imread(img_file)
    img = cv2.resize(img, img_shape, interpolation=cv2.INTER_CUBIC)
    group_images[:, :, i] = img
img90 = np.expand_dims(np.percentile(group_images, 90, axis=-1), axis=-1)
img50 = np.expand_dims(np.percentile(group_images, 50, axis=-1), axis=-1)
img10 = np.expand_dims(np.percentile(group_images, 10, axis=-1), axis=-1)
img = np.concatenate((img10, img50, img90), axis=-1)
img = np.expand_dims(img, axis=0)

fea_vgg16 = xception_model.predict_on_batch(vgg16.preprocess_input(img))
fea_resnet50 = resnet50_model.predict_on_batch(resnet.preprocess_input(img))
fea = np.concatenate((fea_vgg16, fea_resnet50), axis=1)
np.save("fea.npy", fea)

feats = []
feats.append(fea)
arr = np.squeeze(fea)


t0 = time.time()
classes = model.predict(fea)
t1 = time.time()
print("pred time: ", t1-t0)

print("Confidence:", np.max(classes))
print("Label:", labels[np.argmax(classes)])

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(saveimg, labels[np.argmax(classes)] + " " + str(np.max(classes)), (20,50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
cv2.imwrite("./data/" + 'pred.jpg', saveimg)

# upload image to GCP
img_path = os.path.basename(os.path.normpath(img_path))
upload_blob("forabot-web", "./data/pred.jpg", f"pred-{img_path}.jpg")
