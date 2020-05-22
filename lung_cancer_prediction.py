import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('lung_cancer_examples.csv')
 
df.head()

df=df.drop(['Name','Surname'],axis=1)

X=df.iloc[:,:-1].values
y=df.iloc[:,4:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()

X_train = Sc_X.fit_transform(X_train)
X_test= Sc_X.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
DecisionTree_model= DecisionTreeClassifier()

from sklearn.linear_model import LogisticRegression
LogisticReg_model= LogisticRegression(random_state=0)
LogisticReg_model.fit(X_train,y_train)

y_pred= LogisticReg_model.predict(X_test)

LG_acc= accuracy_score(y_test,y_pred)
 
DecisionTree_model.fit(X_train,y_train)

y_pred = DecisionTree_model.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score
classification_report(y_test,y_pred)
DT_acc=accuracy_score(y_test,y_pred)

#sns.scatterplot(x=df['Smokes'],y=df['Alkhol'],hue=df['Result'])
#
#sns.scatterplot(x=df['Age'],y=df['Alkhol'],hue=df['Result'])
#
#sns.scatterplot(x=df['AreaQ'],y=df['Alkhol'],hue=df['Result'])
#
#sns.scatterplot(x=df['Smokes'],y=df['Age'],hue=df['Result'])

sns.kdeplot(df)

sns.pairplot(df,hue='Result', size=2.5)



