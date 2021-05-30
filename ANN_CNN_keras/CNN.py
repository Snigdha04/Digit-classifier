from keras.models import Sequential
from keras import losses
from keras import optimizers
import numpy
import pickle
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical  
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

x, y = pickle.load(open("train.pkl", "rb"))
y = [[p] for p in y]

enc = OneHotEncoder()
enc.fit(y)
y = [enc.transform([p]).toarray() for p in y]
w = []
for yy in y:
    w.append(yy[0])

y = numpy.array(w)
x = x.reshape(-1,28,28,1)

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5, 5),padding='Same',activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(loss=losses.mean_squared_error, optimizer=optimizers.SGD(lr=3.0))
model.fit(x, y, epochs=10, batch_size=60)

x, y = pickle.load(open("test.pkl", "rb"))
x = x.reshape(-1,28,28,1)
classes = model.predict(x)

q = []
for c in classes:
    q.append(numpy.argmax(c))

dif = 0
tot = len(y)
for i in range(len(y)):
    if (y[i] != q[i]): dif += 1

print((tot - dif) / tot)
