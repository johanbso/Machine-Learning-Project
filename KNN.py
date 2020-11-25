# Import the KNN classifier and the performance measure accuracy_score from the Scikit-learn library.
from sklearn.neighbors import KNeighborsClassifier
import Results
import operator
import pandas as pd


"""
Train an optimal K - Nearest Neighbors classifier by testing all combinations of neighbors and weights.
@:param x_train - Training feature value vector
@:param y_train - Training target value vector
@:param x_test - Test feature value vector
@:param y_test - Test target value vector
@:return optimal_classifier - Optimized K - Nearest Neighbors classifier
"""
def knn_classifier(x_train, y_train, x_test,  y_test):
    classifier_list = []
    weights = ['uniform', 'distance']
    metrics2 = ['euclidean', 'manhattan', 'chebyshev']
    ac_list = []

    attributelist = []
    for w in weights:
        for neighbors in range(2, 20):
            for metric in metrics2:
                classifier = KNeighborsClassifier(n_neighbors=neighbors, metric=metric, weights=w)
                classifier.fit(x_train, y_train)
                predictedprobabilities = classifier.predict_proba(x_train)
                cm, ac = Results.accuracy(classifier, x_test, y_test)
                ac_list.append(ac)
                classifier_list.append(classifier)
                attributelist.append([ac, w, neighbors, metric])
    sortedonacc = sorted(attributelist, key=operator.itemgetter(0), reverse=True)
    df = pd.DataFrame(sortedonacc)
    df.columns = ["Accuracy", "Weights", "Neighbours", "Distance metric"]
    df.sort_values(by=['Accuracy'], ascending = (False))
    #print('Prob:  ', predictedprobabilities)
    print("KNN table:")
    print(df.to_latex(index=False))
    index = ac_list.index(max(ac_list))
    optimal_classifier = classifier_list[index]
    return optimal_classifier
