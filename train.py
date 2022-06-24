from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
import argparse
from neuralnet import *
from utility import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--static', action="store_true")
parser.add_argument('--motion', action='store_true')
options = parser.parse_args()

#Test with the svm as well as getting probablities. Issue is that svm
#generates domains within which to classify. Moreover, probabliity will be dictated
#By adheredne to the regression lines, which are pretty limiting. 
#Under that model, regresison lines would have to fit the features, and not necessarily be linear
#It'd require some tweaking 
#

X = []
Y = []

if options.static:
    typ = "static"
elif options.motion:
    typ = "motion"
else:
    print("No svm type given")
    exit

with open("data/"+typ + "collated.txt", "r") as f:
    for l in f.readlines():
        temp = []
        classification = ""
        line = l.split(",")
        for i in range(len(line)-1):
            temp.append(float(line[i].strip()))
        classification = line[-1].strip()
        X.append(temp)
        Y.append(classification)

X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.8, random_state=0)


def svm():
    clf = svm.SVC(decision_function_shape='ovo', probability=True)
    clf.fit(X_train,y_train)
    mm = clf.predict(X_test)
    acc = clf.score(X_test, y_test)

    #Have to tweak once more data is gathered
    tm = svm.SVC(decision_function_shape='ovo', C=5)
    tm.fit(X_train,y_train)
    tacc=tm.score(X_test,y_test)

    correct = []
    for i in range(len(X_test)):
        if mm[i] == y_test[i]:
            correct.append("YES")
        else:
            correct.append("NO")

    print("Accuracy: " + str(acc))
    print("Other Acc: " + str(tacc))
    #print(mm)
    #print("Comparison: ")
    #print(correct)

    with open('models/'+typ+'.pkl', 'wb') as f:
        pickle.dump(clf,f)
        
def nn():
    if typ == "static":
        model = buildStatic(getStaticClasses())
    else:
        pass
    yT= staticEncode(y_train)
    yV = staticEncode(y_test)

    model.fit(X_train, yT.tolist(), epochs=75, batch_size=10)
    _, accuracy = model.evaluate(X_test, yV.tolist())
    
    prediction = model.predict([X_test[3]])[0]
    #Find the index of the largest value (probability) and feed into decoder
    predVal = np.argmax(prediction)
    interpret = decodeStatic([predVal])
    
    print("Accuracy: " + str(accuracy))
    print("Prediction: " + str(interpret))
    print("Prediction probability: " + str(prediction[predVal]))
    print("Actual: " + y_test[3])
    model.save("models/staticnet")
    
nn()