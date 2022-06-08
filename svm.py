from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
   
   
   
X = []
Y = []

with open("data/collated.txt", "r") as f:
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
        
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train,y_train)


mm = clf.predict(X_test)

acc = clf.score(X_test, y_test)

correct = []
for i in range(len(X_test)):
    if mm[i] == y_test[i]:
        correct.append("YES")
    else:
        correct.append("NO")

print(mm)
print("Accuracy: " + str(acc))
print("Comparison: ")
print(correct)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf,f)