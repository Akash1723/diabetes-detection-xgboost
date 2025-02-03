import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df=pd.read_csv("data/diabetes.csv")
y=df['Outcome']
X=df.drop(columns='Outcome')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

pipe=Pipeline([('scale',StandardScaler()),('model',XGBClassifier(objective='binary:logistic',eval_metric='error',learning_rate=0.1,max_depth=5,n_estimators=10))])

pipe.fit(X_train,y_train)
best_pred=pipe.predict(X_test)
acc_sc=accuracy_score(best_pred,y_test)
print(f"Accuracy of the model: {acc_sc * 100:.2f}%")

print(classification_report(y_test, best_pred))

cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, fmt = 'd', annot = True)