# KNN.py
# Author: Alexander Day

# Import relevant libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Given an input confusion matrix, the function will plot the confusion 
    matrix using a colormap visualization
    '''
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def createRF(n_estimators, max_features):
    '''
    Given a n_estimators value, the function will return
    a Random forest model that can be called elsewhere in the script
    '''
    print(' Creating RF model with {} n_estimators, {} max_features'.format(
                                                n_estimators, max_features))
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_features=max_features)
    return model

def optimizeRF(X, Y, maxN=10):
    '''
    Given X and y data, the function will optimize the n_estimator and 
    max_feature best suited to maximizing the RF model accuracy

    Returns [bestScore, bestN, bestMF, bestScaler]
    '''
    print('Trying to optimize RF model...')

    sscaler = StandardScaler()
    mscaler = MinMaxScaler()

    sscaler.fit(X)
    mscaler.fit(X)

    sx = sscaler.transform(X)
    mx = mscaler.transform(X)

    dependents = [X, sx, mx]
    dependentNames = ['None', 'StandardScaler', 'MinMaxScaler']

    # For this model, we just need to use 2 for loops to optimize the 
    # number of neighbors and weight type 
    
    # We set up variables that will be optimized:
    bestN = None
    bestMF = None
    bestI = None
    bestScore = 0.0

    # Our outer loop will optimize K value, while the inner one will
    # optimize weight type
    for i in range(0, 3):
        for n in range(1, maxN):
            for mf in range(1, 30):
                model = createRF(n, mf)
                x = dependents[i]
                # score = cross_val_score(model, x, Y, cv=3).mean()
                score = cross_val_score(model, x, Y, cv=loo).mean()
                if score > bestScore:
                    bestN = n
                    bestMF = mf
                    bestI = i
                    bestScore = score
    
    print('Best score: {}'.format(bestScore))
    print('Best N: {}'.format(bestN))
    print('Best Scaler: {}'.format(dependentNames[bestI]))
    print('Best MF: {}'.format(bestMF))
    
    # Return a list containing the top accuracy and the optimized params:
    return [bestScore, bestN, bestMF, dependentNames[bestI]]

if __name__ == "__main__":
    print('Trying to run RF.py as main')

    dataset = pandas.read_csv(r'Datasets\Iris.csv',
                        header=0)
    dataFrame = pandas.DataFrame(dataset)

    y = dataFrame['Species']
    x = dataFrame.drop(columns=['Id', 'Species'])
    x = x.values

    optimizeRF(x, y)