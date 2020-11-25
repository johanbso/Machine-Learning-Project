import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix

"""
Analyze the accuracy of a classifier.
@:param classifier - Classifier to be analyzed
@:param x_test - Test feature value vector
@:param y_test - Test target value vector
@:return classifier - The fraction of correctly classified samples.
"""
def accuracy(classifier, x_test, y_test):
    # Predicting the test set
    y_pred = classifier.predict(x_test)
    # Making Confusion Matrix and calculating accuracy score
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)

    return cm, ac

"""
Presents the perfomance of the methods in diagram
@:param methodNames - String vector containing the names of the methods
@:param performances - Double vector containing the accuracy scores of the respective methods
"""
def PresentResults(methodNames, perfomances):
    plt.rcParams['figure.figsize']=15,6
    sns.set_style("darkgrid")
    ax = sns.barplot(x=methodNames, y=perfomances, palette = "rocket", saturation =1.5)
    plt.xlabel("Classifier Models", fontsize = 20 )
    plt.ylabel("% of Accuracy", fontsize = 20)
    plt.title("Accuracy of the Classifier Models", fontsize = 20)
    plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
    plt.yticks(fontsize = 13)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
    plt.show()

"""
Presents the perfomance of the methods in a ROC plot
@:param x_test - Test features
@:param performances - Test target values
@:param models - Classifiers
@:param ann - Artificial Neural Network
"""
def roc_plot(x_test, y_test, x_test_pca, y_test_pca, models, models_pca):
    names = ['KNN', "SVM"]
    names_pca = ['KNN (PCA)', "SVM (PCA)"]
    count = 0

    plt.title('Receiver Operating Characteristic')
    # Plot the models not using PCA
    for model in models:
        probs = model.predict_proba(x_test)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        # method I: plt
        plt.plot(fpr, tpr, label='AUC ' + names[count] + ' = %0.2f' % roc_auc)
        count = count + 1

    count = 0
    # Plot the models using PCA
    for model in models_pca:
        probs = model.predict_proba(x_test_pca)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test_pca, preds)
        roc_auc = metrics.auc(fpr, tpr)
        # method I: plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, label='AUC ' + names_pca[count] + ' = %0.2f' % roc_auc)
        count = count + 1

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
