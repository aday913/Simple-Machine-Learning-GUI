# KNN.py
# Author: Alexander Day

# Import relevant libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
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

def createKNN(K, weight):
    '''
    Given a K-neighbor value, the function will return
    a KNN model that can be called elsewhere in the script
    '''
    print(' Creating a KNN model with {} neighbors and {} weight'.format(
                                                                K, weight))
    model = KNeighborsClassifier(n_neighbors=K, weights=weight)
    return model

def optimizeKNN(X, Y, maxK=10):
    '''
    Given X and y data, the function will optimize the K-neighbor and 
    weight best suited to maximizing the KNN model accuracy

    Returns [bestScore, bestK, bestWeight]
    '''

    print('Trying to optimize KNN model...')

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
    bestX = None
    bestK = None
    bestWeight = None
    bestI = None
    bestScore = 0.0

    # Our outer loop will optimize K value, while the inner one will
    # optimize weight type
    for i in range(0, 3):
        for n in range(1, maxK):
            for weight in ['uniform', 'distance']:
                model = createKNN(n, weight)
                x = dependents[i]
                score = cross_val_score(model, x, Y, cv=3).mean()
                if score > bestScore:
                    bestX = x
                    bestK = n
                    bestWeight = weight
                    bestI = i
                    bestScore = score
    
    print('Best score: {}'.format(bestScore))
    print('Best K: {}'.format(bestK))
    print('Best Scaler: {}'.format(dependentNames[bestI]))
    print('Best Weight Type: {}'.format(bestWeight))
    
    # Return a list containing the top accuracy and the optimized params:
    # return [bestScore, bestK, bestWeight, dependentNames[bestI]]

    bestModel = KNeighborsClassifier(n_neighbors=bestK, weights=bestWeight)
    prediction = cross_val_predict(bestModel, bestX, Y, cv=3)
    confusion = confusion_matrix(Y, prediction)
    plot_confusion_matrix(confusion)

if __name__ == "__main__":
    print('Trying to run KNN.py as main')

    dataset = pandas.read_csv(r'Datasets\Iris.csv',
                        header=0)
    dataFrame = pandas.DataFrame(dataset)

    y = dataFrame['Species']
    x = dataFrame.drop(columns=['Id', 'Species'])
    x = x.values

    optimizeKNN(x, y)