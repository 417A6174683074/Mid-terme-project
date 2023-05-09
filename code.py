import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from joblib import dump
import matplotlib.pyplot as plt

titanic=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
verif=pd.read_csv("gender_submission.csv")

titanic.drop('PassengerId',axis=1,inplace=True)
titanic['Cabin'].fillna(value='0',inplace=True)

test.drop('PassengerId',axis=1,inplace=True)
test['Cabin'].fillna(value='0',inplace=True)


rep={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20}



for i in range(len(titanic['Cabin'])):
    if titanic['Cabin'][i]=='0':
        titanic.loc[i,'Cabin']=0
    else:
        a=titanic['Cabin'][i][0]
        titanic.loc[i,'Cabin']=rep[a]
        
        
for i in range(len(test['Cabin'])):
    if test['Cabin'][i]=='0':
        test.loc[i,'Cabin']=0
    else:
        a=test['Cabin'][i][0]
        test.loc[i,'Cabin']=rep[a]

titanic.dropna(inplace=True)
titanic['Cabin']=titanic['Cabin'].astype(int)


titanic.drop('Name',axis=1,inplace=True)
titanic.drop('Ticket',axis=1,inplace=True)



test['Cabin']=test['Cabin'].astype(int)


test.drop('Name',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)


nonint=['Sex','Age','Fare','Embarked']

_clear=pd.get_dummies(titanic[nonint])
titan=pd.concat([titanic,_clear],axis=1)
titan.drop(nonint,axis=1,inplace=True)
titan=titan.astype('int64')

_clear=pd.get_dummies(test[nonint])
test_=pd.concat([test,_clear],axis=1)
test_.drop(nonint,axis=1,inplace=True)
test_=test_.astype('int64')


y=titan['Survived']
x=titan.drop('Survived',axis=1)
verif.drop('PassengerId',axis=1)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['none', 'l2']}

log=LogisticRegression()

clf = GridSearchCV(log, param_grid, cv=5, scoring='accuracy')

clf.fit(x,y)

verif=verif['Survived']

pred=clf.predict(test_)
print(accuracy_score(pred,verif))
print(confusion_matrix(pred, verif))
print(clf.best_params_)
