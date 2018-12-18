import csv
import numpy as np

xTr = []
yTr = []

xTe = []

with open('train.csv', 'r') as train:
    reader = csv.reader(train)
    next(reader)
    for row in reader:
        xTrRow = []
        xTrRow.append(float(row[2]))
        if (row[4] == 'male'):
            xTrRow.append(1.0)
        else:
            xTrRow.append(0.0)
        if (row[5] == ''):
            if (row[1] == '1'):
                xTrRow.append(28.0)
            else:
                xTrRow.append(31.0)
        else:
            xTrRow.append(float(row[5]))
        xTrRow.append(float(row[6]))
        xTrRow.append(float(row[7]))
        xTrRow.append(float(row[9]))
        if (row[11] == 'S'):
            xTrRow.append(1.0)
        elif (row[11] == 'Q'):
            xTrRow.append(2.0)
        else:
            xTrRow.append(3.0)
        yTr.append(int(row[1]))
        xTr.append(xTrRow)

from sklearn.svm import SVC
classifier = SVC().fit(xTr[:800], yTr[:800])
yTe = classifier.predict(xTr[800:])

with open('submission.csv', mode='w') as submission:
	writer = csv.writer(submission)
	for i in range(91):
		writer.writerow([str(i+801), yTe[i]])
