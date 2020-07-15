import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import glob
from collections import Counter

filenames = glob.glob('nonspam*.txt')
nonspam = pd.concat([pd.read_csv(f, header=None) for f in filenames])
nonspam['Spam'] = 0
filenames = glob.glob('spam*.txt')
spam = pd.concat([pd.read_csv(f, header=None) for f in filenames])
spam['Spam'] = 1
data = pd.concat([nonspam, spam])
data = data.rename(columns={0: 'Emails'})

Vocab = []

for email in data["Emails"]:
    Vocab += email.split()

Vocab = Counter(Vocab).most_common(3000)
features = np.empty((960, 3000))
row = 0
col = 0
for email in data["Emails"]:
    mail = email.split()
    for word in Vocab:
        count = (mail.count(word[0]))
        features[row, col] = 1 if count > 0 else 0
        col = col + 1
    col = 0
    row = row + 1


def getparameters(classtype):
    count_nonspam = 0
    count_spam = 0
    Type_zero = np.zeros(3000)
    Type_one = np.zeros(3000)
    Type0 = 0
    Type1 = 0
    for i in data["Spam"]:
        if i == 0:
            count_nonspam += 1
        else:
            count_spam += 1

    Type1 = count_spam / 960
    Type0 = count_nonspam / 960
    feature = features.T
    for email in range(3000):
        num_zero = 0
        num_one = 0

        for output in data["Spam"]:
            col = 0
            x = feature[email, col]
            if (x== 1 and output == 0):
                num_zero += 1

            if (x == 1 and output == 1):
                num_one += 1
            col += 1
        Type_zero[i] = (num_zero + 1) / (1 + count_nonspam)
        Type_one[i] = (num_one + 1) / (1 + count_spam)
    if classtype == 1:
        return Type_one, Type_zero, Type1
    else:
        return Type_one, Type_zero, Type0


def predict(Type_zero, Type_one, Type, X):
    np.seterr(divide='ignore', invalid='ignore')
    for j in range(960):
        feature = X.T
        featuure = feature[j]
        num = np.ones(3000)
        den = np.ones(3000)

        for feature in featuure:

            if (feature == 0):
                num = np.dot(num, 1 - Type_one)
                den = np.dot(den, 1 - Type_zero)
            else:
                num = np.dot(num, Type_one)
                den = np.dot(den, Type_zero)
        num = np.dot(num, Type)
        den = np.dot(den, (1 - Type))
        p = np.sum(np.divide(num, np.add(den, num)))

        print("email: ", j)
        print(p)


kfold = KFold(8, True, 1)
No = 1
Type_one, Type_zero, Type = getparameters(0)
for train, test in kfold.split(features):
    print("TEST SET ", No)
    Type_one, Type_zero, Type = getparameters(0)
    predict(Type_zero,Type_one,Type, features)
    No +=1
