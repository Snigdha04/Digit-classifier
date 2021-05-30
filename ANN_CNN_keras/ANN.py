from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers
import numpy
import pickle
from sklearn.preprocessing import OneHotEncoder

x, y = pickle.load(open("train.pkl", "rb"))
y = [[p] for p in y]

enc = OneHotEncoder()
enc.fit(y)
y = [enc.transform([p]).toarray() for p in y]
w = []
for yy in y:
	w.append(yy[0])

y = numpy.array(w)

model = Sequential()
model.add(Dense(units=30, activation='sigmoid', input_dim=784))
model.add(Dense(units=10, activation='sigmoid'))
model.compile(loss=losses.mean_squared_error, optimizer=optimizers.SGD(lr=3.0))
model.fit(x, y, epochs=30, batch_size=10)

x, y = pickle.load(open("test.pkl", "rb"))

classes = model.predict(x)

q = []
for c in classes:
	q.append(numpy.argmax(c))

dif = 0
tot = len(y)
for i in range(len(y)):
	if(y[i]!=q[i]): dif+=1

print((tot-dif)/tot)