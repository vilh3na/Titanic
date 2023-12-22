import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn. impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Load the dataset
titanic_data = pd.read_csv('data/train.csv')

# Splitting the dataset using StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 2)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1, 2, 1)

strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

#plt.show()

strat_train_set.info()

class AgeImputer (BaseEstimator, TransformerMixin):
    def fit (self, X, y=None):
        return self
    def transform (self, X):
        imputer = SimpleImputer (strategy="mean" )
        X['Age'] = imputer.fit_transform (X[['Age']])
        return X
    
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit (self, X, y=None):
        return self
    def transform (self, X):
        encoder=OneHotEncoder()
        matrix=encoder.fit_transform(X[['Embarked']]).toarray()
        column_names=["C","S","Q","N"]
        for i in range(len(matrix.T)):
            X[column_names[i]]=matrix.T[i]
        matrix=encoder.fit_transform(X[["Sex"]]).toarray()
        column_names=["Female","Male"]
        for i in range (len(matrix.T)):
            X[column_names[i]]=matrix.T[i]
        return X

class FeatureDropper (BaseEstimator,TransformerMixin):
    def fit (self, X, y=None):
        return self
    def transform (self, X):
        return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"], axis=1, errors="ignore")
    
    
pipeline= Pipeline([
    ("ageimputer", AgeImputer()),
    ("featureencoder", FeatureEncoder()),
    ("featuredropper", FeatureDropper())
    ])

strat_train_set=pipeline.fit_transform(strat_train_set)

strat_train_set.info()

X=strat_train_set.drop(['Survived'], axis=1)
y=strat_train_set['Survived']
scaler= StandardScaler()
X_data = scaler.fit_transform(X)
y_data= y.to_numpy()

clf=RandomForestClassifier ()

param_gird= [
    {"n_estimators":[10,200,500,800],"max_depth": [None, 5, 10],"min_samples_split":[2,3,4]}
]
grid_search= GridSearchCV(clf,param_gird,cv=3,scoring="accuracy",return_train_score=True)
grid_search.fit(X_data,y_data)

final_clf=grid_search.best_estimator_

print(final_clf)

strat_test_set=pipeline.fit_transform(strat_test_set)

X_test=strat_test_set.drop(["Survived"],axis=1)
y_test=strat_test_set['Survived']

scaler=StandardScaler()
X_data_test=scaler.fit_transform(X_test)
y_data_test=y_test.to_numpy()
print(final_clf.score(X_data_test,y_data_test))

final_data = pipeline.fit_transform(titanic_data)
print(titanic_data)

X_final=final_data.drop(["Survived"],axis=1)
y_final=final_data['Survived']

scaler=StandardScaler()
X_data_final=scaler.fit_transform(X_final)
y_data_final=y_final.to_numpy()

prod_clf=RandomForestClassifier ()

param_gird= [
    {"n_estimators":[10,200,500,800],"max_depth": [None, 5, 10],"min_samples_split":[2,3,4]}
]
grid_search= GridSearchCV(prod_clf,param_gird,cv=3,scoring="accuracy",return_train_score=True)
grid_search.fit(X_data_final,y_data_final)

prod_final_clf=grid_search.best_estimator_
print(prod_final_clf)
print(prod_final_clf.score(X_data_final,y_data_final))

titanic_test_data = pd.read_csv("data/test.csv")
final_test_data= pipeline.fit_transform(titanic_test_data)
final_test_data.info
print(final_test_data)
print('cima')

X_final_test =final_test_data
X_final_test= X_final_test.fillna(method="ffill")

scaler=StandardScaler()
X_data_final_test = scaler.fit_transform(X_final_test)

predictions = prod_final_clf.predict(X_data_final_test)
final_df=pd.DataFrame(titanic_test_data['PassengerId'])
final_df['Survived'] = predictions
final_df.to_csv("data/predictions.csv",index=False)
