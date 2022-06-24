from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



def buildStatic(classes):
    model = Sequential()
    model.add(Dense(126, input_shape=(63,), activation='relu'))
    model.add(Dense(126, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(63, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
