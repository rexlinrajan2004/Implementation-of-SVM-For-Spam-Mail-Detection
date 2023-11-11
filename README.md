# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: REXLIN R
RegisterNumber:  212222220034
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Data Head:
![image](https://github.com/rexlinrajan2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119406566/3e3f4271-a6c3-49bb-9936-730d42a3e720)
## Data Info:
![image](https://github.com/rexlinrajan2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119406566/193edd0f-c6f9-419a-a816-8699e98ddcc4)
## Data isnull():
![image](https://github.com/rexlinrajan2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119406566/4216195e-44eb-4238-98fa-8956aca2b0bc)
## y_pred:
![image](https://github.com/rexlinrajan2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119406566/a21733be-5292-4c6f-b739-8c7c9900e936)
## Accuracy:
![image](https://github.com/rexlinrajan2004/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119406566/7e843899-38d5-412b-9701-fcdbbe06b68d)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
