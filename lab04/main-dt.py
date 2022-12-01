##Authors: Karol Kuchnio s21912 and Michał Wadas s20495
import pandas as pd;
import numpy as np
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score, f1_score
if __name__ == '__main__':

    dataset = pd.read_csv('./data/winequality-white.csv', sep=';')
    """
     Pick the quality as a target variable
    """
   
    y = dataset.quality
    X = dataset.drop('quality', axis=1)
    """
     set reproducibility
    """
    seed = 8
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=seed)
    

    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train, y_train)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    predictions = clf.predict(X_test)
    """
     So now we can predict how our model is predicting: 0 - malignant, 1-benign
    """
    print("Prediction:",predictions)
    """
     Show difference between tree that doesnt have stopping criteria and the one that does
    """
    proba = clf.predict_proba(X_test)
    """
     #this will gave us probability for each of the classes(data point by data point):
     #1. - class 0 with one out of one chance
     #0. - class 1 with zero out of one chance
    """
    print(proba)
    """
      #measure accuracy of this prediction:
    """
    print("Accuracy score metric for prediction: ",accuracy_score(y_test,predictions))

    #
    results = confusion_matrix(y_test, predictions, labels=[range(0,10)])
    print('Additional result from confusion matrix',results )
    print('1 happened',results[0][0] )
    print('2 happened',results[1][1] )
    print('3 happened',results[2][2] )
    print('4 happened',results[3][3] )
    print('5 happened',results[4][4] )
    print('6 happened',results[5][5] )
    print('7 happened',results[6][6] )
    print('8 happened',results[7][7] )
    print('9 happened',results[8][8] )
    print('10 happened',results[9][9] )

    """
      measure accuracy of this prediction:
    """
    print("Precision score metric for prediction: ",precision_score(y_test,predictions, average='weighted'))
   
    classifier = SVC(random_state = 1, kernel = 'rbf')
    classifier.fit(X_train, y_train)
    """
      Predicting Test Set
    """
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred))
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred,average='weighted',labels=np.unique(y_pred))

    results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    print(results)