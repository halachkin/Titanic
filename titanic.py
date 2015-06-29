"""Titanic: Machine Learning from Disaster
https://www.kaggle.com/c/titanic/data"""

import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier


TRAIN_SIZE = 891

def prepare_data():
    """stage zero - preparing data
    """
    #all data to one table
    df = pd.concat( \
        [pd.read_csv('data//train.csv', header=0).drop(['Survived'], axis=1),
         pd.read_csv('data//test.csv', header=0)])
    df = df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis = 1)

    # replace missing fares with median fare
    median_fare = df['Fare'].dropna().median()
    df.loc[(df.Fare.isnull()), 'Fare'] = median_fare

    # replace missing embarked with mode embarked
    mode_embarked = df['Embarked'].dropna().mode()
    df.loc[(df.Embarked.isnull()), 'Embarked'] = mode_embarked[0]
    #map categorical features to numbers
    df['Sex'] = df['Sex'].map({'female': 0.0, 'male': 1.0}).astype(float)
    df['Embarked'] = df['Embarked'].map( \
                            {'C': 0.0, 'S': 1.0, 'Q': 2.0}).astype(float)

    #There are a lot of items with missing age, so we will predict it
    X_train = df[df['Age'].notnull()].drop('Age', axis=1).values
    #add polynomial features
    X_train = PolynomialFeatures(degree=2).fit_transform(X_train)
    #get targets 
    y_train = df[df['Age'].notnull()].Age.values
    return df, X_train, y_train


def predict_age(X_train, y_train, X_test):
    """stage one - predicting age
    """
    #train the model
    #Looks that  Regression is far away from reality, but I hope, that 
    #it still better than medians
    model =  GradientBoostingRegressor()   
    acc = cross_validation.cross_val_score(model, 
                                           X_train, y_train,
                                           cv=5)

    print "================================"
    print "Age:"
    print "5 fold cross-validation R squared:\n" + str(acc) 
    print "mean R squared: " + str(acc.mean())
    print "std dev: "  + str(acc.std())
    model.fit(X_train, y_train)
    print "training data  R squared:"  + str(model.score(X_train,y_train))
    y = model.predict(X_test)
    return y


def fit_classifier(df):
    """stage two - predicting surivival
    """
    #get train data (with age feature)
    X_train = df[0:TRAIN_SIZE].values
    y_train = pd.read_csv('data//train.csv', header=0)['Survived'].values
    #add polynomials to train features
    X_train = PolynomialFeatures(degree=2).fit_transform(X_train)
    #init and test the model
    model = GradientBoostingClassifier(max_depth=1)
    acc = cross_validation.cross_val_score(model, 
                                           X_train, y_train,
                                           scoring='accuracy', 
                                           cv=5)

    print "================================"
    print "Surivival:"
    print "5 fold cross-validation accuracy:\n" + str(acc) 
    print "mean accuracy: " + str(acc.mean())
    print "std dev: "  + str(acc.std())

    #train the model
    model.fit(X_train,y_train)
    print "training data accuracy:"  + str(model.score(X_train,y_train))
    return model


def predict_and_save(model, X_test):
    output = model.predict(X_test)
    predictions_file = open("output.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    ids = pd.read_csv('data//test.csv', header=0)['PassengerId'].values
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()

if __name__ == "__main__":

    #predicting age
    df, X_train, y_train = prepare_data()
    X_test = df[df['Age'].isnull()].drop('Age', axis=1).values
    X_test = PolynomialFeatures(degree=2).fit_transform(X_test)
    age = predict_age(X_train, y_train, X_test)
    df.loc[(df.Age.isnull()), 'Age'] = age
    #predicting survival
    model = fit_classifier(df)
    X_test = df[TRAIN_SIZE:].values
    X_test = PolynomialFeatures(degree=2).fit_transform(X_test)
    predict_and_save(model, X_test)


