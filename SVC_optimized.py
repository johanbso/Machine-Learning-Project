from sklearn.svm import SVC
import Results

"""
Train an optimal K - Nearest Neighbors classifier.
@:param x_train - Training feature value vector
@:param y_train - Training target value vector
@:param x_test - Test feature value vector
@:param y_test - Test target value vector
@:return optimal_classifier - Optimized K - Nearest Neighbors classifier
"""
def svc_classifier(x_train, y_train, x, y):
    c = []
    a = []
    for d in range(1, 10):
        print("training")
        classifier = SVC(kernel='rbf', degree=d, gamma='scale', shrinking=True, C=4.6, probability=True)
        classifier.fit(x_train, y_train)
        a.append(Results.accuracy(classifier, x, y)[1])
        c.append(classifier)
        print("done")

    index = a.index(max(a))
    classifier = c[index]
    return classifier
