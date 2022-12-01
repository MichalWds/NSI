##Authors: Karol Kuchnio s21912 and Michał Wadas s20495
import pandas as pd;
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score
if __name__ == '__main__':
    """
     Decision tree for Brest cancer
    """
    data = load_breast_cancer()

    dataset  = pd.DataFrame(data=data['data'],columns=data['feature_names'])
    """
    #all features you want to give to determin if the data point is benign or malignant4
    """
    X = dataset.copy()
    """
    #that could be 0 or 1 - depends if its good or bad
    """
    y = data['target'] 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    """
    #so now we can predict how our model is predicting: 0 - malignant, 1-benign
    """
    print("Prediction:",predictions)
    """
    #Show difference between tree that doesnt have stopping criteria and the one that does
    """
    proba = clf.predict_proba(X_test)
    """
    #this will gave us probability for each of the classes(data point by data point):
    #1. - class 0 with one out of one chance
    #0. - class 1 with zero out of one chance
    """
    print("Probability:",proba)
    """
    #measure accuracy of this prediction:
    """
    print("Accuracy score metric for prediction: ",accuracy_score(y_test,predictions))

    
    results = confusion_matrix(y_test, predictions, labels=[0,1])
    print('Additional confusion matrix result',results )
    print('0 malignant happened',results[0][0] )
    print('1 benign happened',results[1][1] )
    """
    #measure accuracy of this prediction:
    """
    print("Precision score metric for prediction: ",precision_score(y_test,predictions))