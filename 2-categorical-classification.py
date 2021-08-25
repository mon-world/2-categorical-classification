from PIL import Image
import os, glob, sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import csv

# 이미지 파일 불러오기
img_dir = '/content/drive/MyDrive/Colab Notebooks/시각지능/day16/raw-img'
categories = ['chichen','cat']
np_classes = len(categories)

image_w = 100
image_h = 100

pixel = image_h * image_w * 3

X=[]
y=[]

# 이미지 파일 분류하기
for idx, messy_desk in enumerate(categories) : ## 인덱스 값과 벨류 값이 같이나옴 enumerate
  img_dir_detail = img_dir + "/" + messy_desk
  files = glob.glob(img_dir_detail + "/*") ## 파일들이 리스트 형태로 저장된다.

  for i, f in enumerate(files) :
    try :
      img = Image.open(f)
      img = img.convert("RGB")
      img = img.resize((image_w, image_h))
      data = np.asarray(img)
      # Y는 0 아니면 1 이므로 idx 값으로 넣는다.
      X.append(data)
      y.append(idx)
      if i % 300 == 0 :
        print(messy_desk, ":", f)
      
    except :
      print(messy_desk, str(i) + " 번째에서 에러", f)

X = np.array(X)
y = np.array(y)

## train set 0.9, test set 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

## 전처리 된 파일을 저장한다 : check point 1
xy = (X_train, X_test, y_train, y_test)
np.save("/content/drive/MyDrive/Colab Notebooks/practice/binary_Image_data.npy", xy)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255



# 모델 구축
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.layers import AveragePooling2D
from tensorflow import keras
from tensorflow.keras import layers

# 모델 생성
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape = (100,100,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일 및 학습
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model_dir = '/content/drive/MyDrive/Colab Notebooks/시각지능/day17/model'
if not os.path.exists(model_dir) :
  os.mkdir(model_dir)
model_path = model_dir + "/m_c_classify.model"

checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train, callbacks=[checkpoint, early_stopping], batch_size=50, validation_data = (X_test,y_test), epochs = 100)



# 그래프 생성
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', "accuracy", 'val_accuracy'], loc='upper left')
plt.show()