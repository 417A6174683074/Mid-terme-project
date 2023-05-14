import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, classification_report
import seaborn as sns
from joblib import dump,load
import numpy as np
import matplotlib.pyplot as plt
'''
0: 1-29 days past
1: 30-59 days past
2: 60-89 days past
3:90-119 days past
4: 120-149 days past
5: 150+ days past
C: paid off their loans
X: no loans

'''


rep={'X':1,'C':1,'0':1,'1':1,'2':0,'3':0,'4':0,'5':0}
good=['X','C','0']
bad=['1','2','3','4','5']



cred=pd.read_csv("credit_record.csv")
app=pd.read_csv("application_record.csv")
tot=pd.merge(cred,app,on='ID',how="inner")

tot['OCCUPATION_TYPE'].fillna(value='Unemployed',inplace=True)
tot.drop('ID',axis=1,inplace=True)



tot['STATUS'] = tot['STATUS'].apply(lambda x: 1 if x in good else 0)
total=tot.copy()




nonint=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']

_clear=pd.get_dummies(tot[nonint])
tot.drop(nonint,axis=1,inplace=True)

cred=pd.concat([tot,_clear],axis=1)
cred.drop(['CODE_GENDER_M','FLAG_OWN_CAR_N','FLAG_OWN_REALTY_N',],axis=1,inplace=True)



y=cred['STATUS']
x=cred.drop('STATUS',axis=1)
print('O')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('a')





param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

model = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='accuracy', cv=5)
search.fit(X_train, y_train)

print("best hyperparametres :", search.best_params_)
print("best score :", search.best_score_)

best_model = search.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)




y_pred=model.predict(X_test)
print('d')

dump(model,"credit_card.joblib")


accuracy=accuracy_score(y_test,y_pred)
c=confusion_matrix(y_test,y_pred)
sns.heatmap(c, annot=True, cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("predicted values")
plt.ylabel("real values")
plt.show()


r=recall_score(y_test,y_pred)
print(r)
recall_score=0.993246197315445

approved=total[total["STATUS"]==1]

for column in total.columns:
    if column != 'STATUS':
        
        proportions_approbation = approved[column].value_counts(normalize=True)

    
        plt.bar(proportions_approbation.index, proportions_approbation.values)

    
        plt.xlabel(column)
        plt.ylabel('approbation proportion')
        plt.title(f'approbation proportion by {column}')
        plt.legend(fontsize='small')
        plt.show()
