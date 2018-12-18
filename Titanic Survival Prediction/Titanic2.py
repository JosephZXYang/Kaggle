import csv
import numpy as np

xTr = []
yTr = []

xTe = []

sample_weight_ = []

with open('train.csv', 'r') as train:
    reader = csv.reader(train)
    next(reader)
    for row in reader:
        xTrRow = []
        xTrRow.append(float(row[2]))
        if (row[4] == 'male'):
            xTrRow.append(1.0)
        if (row[4] == 'female'):
            xTrRow.append(0.0)
        if (row[5] == ''):
            if (row[1] == '1'):
                xTrRow.append(28.0)
            else:
                xTrRow.append(31.0)
            sample_weight_.append(0.9)
        else:
            xTrRow.append(float(row[5]))
            sample_weight_.append(1.1)
        xTrRow.append(float(row[6]))
        xTrRow.append(float(row[7]))
        if (row[11] == 'S'):
            xTrRow.append(5.0)
        elif (row[11] == 'C'):
            xTrRow.append(1.0)
        else:
            xTrRow.append(0.0)
        yTr.append(int(row[1]))
        xTr.append(xTrRow)

with open('test.csv', 'r') as test:
    reader = csv.reader(test)
    next(reader)
    for row in reader:
        xTeRow = []
        xTeRow.append(float(row[1]))
        if (row[3] == 'male'):
            xTeRow.append(1.0)
        if (row[3] == 'female'):
            xTeRow.append(0.0)
        if (row[4] == ''):
            xTeRow.append(0.0)
        else:
            xTeRow.append(float(row[4]))
        xTeRow.append(float(row[5]))
        xTeRow.append(float(row[6]))
        if (row[10] == 'S'):
            xTeRow.append(5.0)
        elif (row[10] == 'C'):
            xTeRow.append(1.0)
        else:
            xTeRow.append(0.0)
        xTe.append(xTeRow)

weight = {0:1, 1:1.2}
from sklearn.svm import SVC
classifier = SVC().fit(xTr, yTr, sample_weight_)
yTe = classifier.predict(xTe)

with open('submission.csv', mode='w') as submission:
    writer = csv.writer(submission)
    writer.writerow(['PassengerId','Survived'])
    for i in range(418):
        writer.writerow([str(i+892), yTe[i]])
