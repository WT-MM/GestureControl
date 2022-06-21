import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--collate', type="str", help="Collate 'motion' or 'static' data")


def getRelPositions(landmarks):
    landmarks = landmarks.landmark
    dists = []
    dists.append(calcDist(1,0, landmarks))
    dists.append(calcDist(2,1, landmarks))
    dists.append(calcDist(3,2, landmarks))
    dists.append(calcDist(4,3, landmarks))
    dists.append(calcDist(5,0, landmarks))
    dists.append(calcDist(6,5, landmarks))
    dists.append(calcDist(7,6, landmarks))
    dists.append(calcDist(8,7, landmarks))
    dists.append(calcDist(9,5, landmarks))
    dists.append(calcDist(10,9, landmarks))
    dists.append(calcDist(11,10, landmarks))
    dists.append(calcDist(12,11, landmarks))
    dists.append(calcDist(13,9, landmarks))
    dists.append(calcDist(14,13, landmarks))
    dists.append(calcDist(15,14, landmarks))
    dists.append(calcDist(16,15, landmarks))
    dists.append(calcDist(17,13, landmarks))
    dists.append(calcDist(18,17, landmarks))
    dists.append(calcDist(19,18, landmarks))
    dists.append(calcDist(20,19, landmarks))
    dists.append(calcDist(17,0, landmarks))
    return [item for sublist in dists for item in sublist]

def calcDist(first, second, landmarks):
    return (landmarks[first].x-landmarks[second].x, landmarks[first].y-landmarks[second].y, landmarks[first].z-landmarks[second].z)

#Stuff for motion

def processArray(fps, array):
    arr = array.copy()
    if(fps > 30):
        extra = fps-30
        for i in range(extra):
            arr.pop(random.randrange(len(arr)))
    elif(fps < 30):
        missing = 30-fps
        for i in range(missing):
            num = random.randrange(2,len(arr)-2)
            arr.insert(num, arr[num-1])
    return arr.copy()

def getInitDiff(array):
    newArr = []
    for i in range(len(array)):
        newArr.append(compareInstances(array[0], array[i]))
    return [item for sublist in newArr for item in sublist]
    
def compareInstances(first, second):
    dists = []
    for i in range(21):
        dists.append([second[i].x - first[i].x, second[i].y - first[i].y, second[i].z - first[i].z])
    return [item for sublist in dists for item in sublist]


def collateData(dir, name="collated.txt"):
    files = os.listdir(dir)
    lines = []
    for file in files:
        with open(dir+"/"+file, 'r') as f:
            tLines = f.read().splitlines()
            lines.extend(tLines)
    with open(name, "w") as f:
        for line in lines:
            f.write(line + "\n")

if __name__ == "__main__":
    options = parser.parse_args()
    if options.collate in ['static', 'motion']:
        collateData("data/"+options.collate, name="data/"+options.collate+"collated.txt")
                