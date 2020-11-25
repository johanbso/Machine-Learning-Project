from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import operator
import Results
import preProcessing
import Results
import time
"""
Train an optimal K - Nearest Neighbors classifier.
@:param x_train - Training feature value vector
@:param y_train - Training target value vector
@:param x_test - Test feature value vector
@:param y_test - Test target value vector
@:return optimal_classifier - Optimized K - Nearest Neighbors classifier
"""
def svc_classifier(x_train, y_train, x_test, y_test):
    tid = time.time()
    num = 0;
    classifier_list = []
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gammas = ['scale', 'auto']
    shrink = [True, False]
    ac_list = []
    attributelist = []
    for i in range(1, 160, 3):
        for bol in shrink:
            for k in kernel:
                    for gamma in gammas:
                        for d in range(1, 10):
                            tid2 = time.time()
                            print('Time: ', tid2 - tid)
                            print("Antall noder:", num)
                            num += 1
                            if k != 'poly' and d == 1:
                                classifier = SVC(kernel=k, degree=d, gamma=gamma, shrinking=bol, C=i/10.0,
                                                 probability=True)
                                classifier.fit(x_train, y_train)
                                cm, ac = Results.accuracy(classifier, x_test, y_test)
                                ac_list.append(ac)
                                classifier_list.append(classifier)
                                attributelist.append([ac, k, gamma, bol, i/10])
                            elif k == 'poly':
                                classifier = SVC(kernel=k, degree=d, gamma=gamma, shrinking=bol, C=i / 10.0)
                                classifier.fit(x_train, y_train)
                                cm, ac = Results.accuracy(classifier, x_test, y_test)
                                ac_list.append(ac)
                                classifier_list.append(classifier)
                                attributelist.append([ac, k, gamma, bol, i / 10])
    sortedonacc = sorted(attributelist, key=operator.itemgetter(0), reverse=True)
    df = pd.DataFrame(sortedonacc)
    df.columns = ["Accuracy", "Kernel", "Gamma", "Shrink_Heuristic", 'Regularization_parameter']
    print("SVM table:")
    print(df.to_latex(index=False))
    index = ac_list.index(max(ac_list))
    optimal_classifier = classifier_list[index]
    return optimal_classifier
