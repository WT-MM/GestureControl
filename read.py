import keras
import numpy
from sklearn.preprocessing import LabelEncoder

model = keras.models.load_model('models/temp')
enc = LabelEncoder()
enc.classes_ = numpy.load("models/tempEncoder.npy")
temp = []
with open("reference/testPredict.txt", "r") as f:
    l = f.readline()
    line = l.split(",")
    print(line)
    for i in range(len(line)):
        temp.append(float(line[i].strip()))

print(enc.inverse_transform(model.predict(numpy.array([temp]))))