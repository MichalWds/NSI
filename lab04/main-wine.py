
import pandas as pd;
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score, f1_score
if __name__ == '__main__':

    dataset = pd.read_csv('./data/winequality-white.csv', sep=';')
    #Pick the quality as a target variable
    y = dataset.quality
    X = dataset.drop('quality', axis=1)
    #set reproducibility
    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=seed)
    
    classifier = SVC(random_state = 0, kernel = 'rbf')
    classifier.fit(X_train, y_train)

    
    # Predicting Test Set
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    results = results.append(model_results, ignore_index = True)
    print(results)

    # to jest output: py main-wine.py
    #                  Model  Accuracy  Precision    Recall  F1 Score
    # 0  Logistic Regression  0.446939    0.49677  0.446939  0.373422
    # 1            SVM (RBF)  0.446939    0.49677  0.446939  0.373422
    
    
    
    
    
    
    
    
    # clf = SVC(random_state = seed)
    # clf = clf.fit(X_train, y_train)

    # predictions = clf.predict(X_test)
    # #so now we can predict how our model is predicting quality of wine
    # print("Prediction:",predictions)
    # #Show difference between tree that doesnt have stopping criteria and the one that does
    # #proba = clf.predict_proba(X_test)
    # #this will gave us propability for each of the classes(data point by data point):
    # #1. - class 0 with one out of one chance
    # #0. - class 1 with zero out of one chance
    # #print("Prob:",proba)
    
    # #measure accuracy of this prediction:
    # accuracy = accuracy_score(y_test,predictions)
    # print("Accuracy score metric for prediction: ",accuracy)

   

    
    # #measure accuracy of this prediction:
    # precision = precision_score(y_test,predictions, average='micro')
    # print("Precision score metric for prediction: ",precision)
    
    # ##recal 
    # rec = recall_score(y_test, predictions)
    # f1 = f1_score(y_test, predictions)
    # #rbf kernel by default
    # classifier = SVC(random_state = 0)
    # classifier.fit(X_train, y_train)
    # model_results = pd.DataFrame([['SVM (RBF)', accuracy, precision, rec, f1]],
    #            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # results = results.append(model_results, ignore_index = True)
    # print(results)
    
    # #
    # results = confusion_matrix(y_test, predictions, labels=[range(0,10)])
    # print('result',results )