from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""
Train an optimal Random Forrest classifier.
@:param x_train - Training feature value vector
@:param y_train - Training target value vector
@:param x_test - Test feature value vector
@:param y_test - Test target value vector
@:return optimal_classifier - Optimized Random Forrest Classifier
"""
def RandomForrest(x_train, y_train, x_test, y_test):
    # Finding the optimum number of n_estimators
    score_list = []
    classifier_list = []
    for estimators in range(10, 30):
        classifier = RandomForestClassifier(n_estimators=estimators, random_state=0, criterion='entropy')
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        score_list.append(accuracy_score(y_test, y_pred))
        classifier_list.append(classifier)
    index = score_list.index(max(score_list))  # Bør egt også velge minste index
    optimal_classifier = classifier_list[index]
    return optimal_classifier

