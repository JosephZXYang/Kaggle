import csv
import numpy as np

xTr = []
yTr = []

xTe = []

sample_weight_ = []

count = {"1_l":0,"1_d":0,"2_l":0,"2_d":0,"3_l":0,"3_d":0,"m_l":0,"m_d":0,"w_l":0,"w_d":0,
        "c_l":0,"c_d":0,"q_l":0,"q_d":0,"s_l":0,"s_d":0,}

a=0

with open('train.csv', 'r') as train:
    reader = csv.reader(train)
    next(reader)
    for row in reader:
        xTrRow = []
        if (row[1] == '1'):
            if (row[2] == '1'):
                count["1_l"] += 1
            if (row[2] == '2'):
                count["2_l"] += 1
            if (row[2] == '3'):
                count["3_l"] += 1
            if (row[4] == 'male'):
                count["m_l"] += 1
            if (row[4] == 'female'):
                count["w_l"] += 1
            if (row[11] == 'S'):
                count["s_l"] += 1
            if (row[11] == 'C'):
                count["c_l"] += 1
            if (row[11] == 'Q'):
                count["q_l"] += 1
        if (row[1] == '0'):
            if (row[2] == '1'):
                count["1_d"] += 1
            if (row[2] == '2'):
                count["2_d"] += 1
            if (row[2] == '3'):
                count["3_d"] += 1
            if (row[4] == 'male'):
                count["m_d"] += 1
            if (row[4] == 'female'):
                count["w_d"] += 1
            if (row[11] == 'S'):
                count["s_d"] += 1
            if (row[11] == 'C'):
                count["c_d"] += 1
            if (row[11] == 'Q'):
                count["q_d"] += 1
        if (row[2] == '1'):
            xTrRow.append(5.0)
        if (row[2] == '2'):
            xTrRow.append(0.0)
        if (row[2] == '3'):
            xTrRow.append(15.0)
        if (row[4] == 'male'):
            xTrRow.append(5.0)
        if (row[4] == 'female'):
            xTrRow.append(0.0)
        if (row[5] == ''):
            if (row[1] == '1'):
                xTrRow.append(28.0/50)
            else:
                xTrRow.append(31.0/50)
            sample_weight_.append(0.9)
        else:
            xTrRow.append(float(row[5])/50)
            sample_weight_.append(1.1)
        xTrRow.append(float(row[6])/5)
        xTrRow.append(float(row[7])/5)
        if (row[11] == 'S'):
            xTrRow.append(5.0)
        elif (row[11] == 'C'):
            xTrRow.append(1.0)
        else:
            xTrRow.append(0.0)
        yTr.append(int(row[1]))
        xTr.append(xTrRow)
        if (len(xTrRow) != 6):
            a += 1

#weight = {0:1, 1:1.1}
print(count)
print(a)
print(len(xTr) == len(yTr))
from sklearn.svm import SVC
classifier = SVC().fit(xTr[:800], yTr[:800], sample_weight_[:800])
yTe = classifier.predict(xTr[800:])
print(classifier)

with open('submission.csv', mode='w') as submission:
	writer = csv.writer(submission)
	for i in range(91):
		writer.writerow([str(i+801), yTe[i]])
