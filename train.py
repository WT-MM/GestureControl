import os
from pyexpat import model
from unittest import mock
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
import numpy
import keras



    
    
def baseline_model():
    model = Sequential()
    #63 input nodes, 126 hidden nodes
    model.add(Dense(126, input_dim=63, activation='relu'))
    model.add(Dense(63, input_dim=126, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
def main():
    X = []
    Y = []


    with open("data/collated.txt", "r") as f:
        for l in f.readlines():
            temp = []
            classification = ""
            line = l.split(",")
            for i in range(len(line)-1):
                temp.append(float(line[i].strip()))
            classification = line[-1]
            X.append(temp)
            Y.append(classification)
        
    X = numpy.array(X)
    Y = numpy.array(Y)

    enc = LabelEncoder()
    enc.fit(Y)

    #estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
    model = baseline_model()


    encodedY = enc.transform(Y)
    print(encodedY)
    model.fit(
        x=X,
        y=encodedY,
        batch_size=32,
        epochs=250
    )

    model.save("models/temp")
    numpy.save('models/tempEncoder.npy', enc.classes_)


    #To read data bacK: enc.inverse_transform(output_data)
    #Input is 21 * 3 = 63

if __name__ == '__main__':
    main()