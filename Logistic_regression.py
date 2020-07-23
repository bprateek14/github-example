import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_excel('Dataset_Question2.xlsx')
X = np.array(df.drop('Test',axis=1))
Y = np.array(df['Test'].replace('Pass',1).replace('Fail',0))

def sig(x):
    a = (1/(1+ (np.exp(-x))))
    return a

def out_func(x,weight,b):
    z = sig(np.dot(x,weight) + b)
    return z

def error(y,y_hat):
    err = (np.sum((y-y_hat)**2))/(2*y.size)
    return err

def update_weight(x,y,weight,b,learnRate):
    output = out_func(x,weight,b)
    grad = (y-output)*x
    weight += learnRate*grad
    b += learnRate*(y-output)
    return weight,b

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

iteration = int(input('epochs: '))# epochs
learnRate = 0.01 # learning rate

# assigning trainig data
features = x_train 
target = y_train

err = []
#for j in range(10):
# number of records and the number of features
n_records, n_features = features.shape

# initiating random weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
#weights = [-1.92,-255.289,-93.3165,-328.5956,75.9408]

print('Initial Random Weights: ',weights[:n_features])

#starting with bias 0
b = 0

for i in range(iteration):
    for x,y in zip(features,target):
        output = out_func(x,weights,b)
        er = error(y,output)
        weights, b = update_weight(x,y,weights,b,learnRate)
    # now since weights have been updated it's time to calculate output and calculate error  
    out = out_func(features,weights,b)
    loss = np.mean(error(target,out))
    err.append(loss)
    if i%10 == 0:
        print('\n-------------------------')
        prediction = out > 0.5
        accuracy = np.mean(prediction.astype('int') == target)
        print("Train Accuracy: ", accuracy)
             

out1 = out_func(x_test,weights,b).astype('int')
accuracy1 = np.mean(out1==y_test)

print("Test Accuracy: ", accuracy1*100,'%','\n')
print('confusion matix: \n',confusion_matrix(y_test,out1),'\n')
print('classification Report: \n',classification_report(y_test,out1),'\n')

plt.plot(err)
