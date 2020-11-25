# Import classifiers:
from sklearn.metrics import plot_confusion_matrix

from KNN import knn_classifier
from SVC import svc_classifier
from Results import *
from preProcessing import *

"""Data"""
# Get data from CSV-file
dataFrame = getData('data.csv')

# Set target variable and map the target variable from string to numerical value
targetvar = 'diagnosis'
dataFrame[targetvar] = dataFrame[targetvar].map({'M': 0, 'B': 1})

# Remove not useful columns
dataFrame.drop(['id', 'Unnamed: 32'], inplace=True, axis=1) #Fjern inplace=True

# Manipulate the data set (Moving target column to be the last column)
cols = dataFrame.columns.tolist()
cols.remove(targetvar)
cols.append(targetvar)
dataFrame = dataFrame[cols]

# Information about the data set
dataFrame.describe()

# Plot histograms
#violin_plot(dataFrame)

# Find and plot the optimal numbers of features, and select the most important ones
feat = rfe(dataFrame)
dataFrame = dataFrame[feat]

# Feature importance using extra trees classifier
# feat2 = featureImportance(dataFrame)
# dataFrame2 = featureReduction(dataFrame, feat[:-2])

# Create boxplot for the features that is selected
# boxplot(dataFrame)

# Principal Component Analysis(Data is normalized in the PCA method)
dataFrame_pca = pca(dataFrame, targetvar)

# Pairplot the principal components
# runPairPlot(dataFrame)

# Split data into 80% training and 20% testing
x_train, x_test, y_train, y_test = divideData(dataFrame)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = divideData(dataFrame_pca)

# Standardize the data (not the PCA data)
x_train, x_test = featureScaler(x_train, x_test)

# Train an optimal KNN classifier
knn = knn_classifier(x_train, y_train, x_test, y_test)
knn_pca = knn_classifier(x_train_pca, y_train_pca, x_test_pca, y_test_pca)

# Train an optimal SVM classifier
svm = svc_classifier(x_train, y_train, x_test, y_test)
"""svm_pca = svc_classifier(x_train_pca, y_train_pca, x_test_pca, y_test_pca)"""

# Accuracy and Confusion Matrix
knn_cm, knn_ac = accuracy(knn, x_test, y_test)
knn_pca_cm, knn_pca_ac = accuracy(knn_pca, x_test_pca, y_test_pca)
svm_cm, svm_ac = accuracy(svm, x_test, y_test)
"""svm_cm_pca, svm_ac_pca = accuracy(svm_pca, x_test_pca, y_test_pca)"""

# Plot accuracy of the models
"""
methods = ["KNN", "KNN (PCA)", "SVM", "SVM (PCA)"]
performances = [knn_ac, knn_pca_ac, svm_ac, svm_ac_pca]
"""
methods = ["KNN", "KNN (PCA)", "SVM"]
performances = [knn_ac, knn_pca_ac, svm_ac]
PresentResults(methods, performances)

# Plot the confusion matrix for the KNN classifier
plot_confusion_matrix(knn, x_test, y_test, cmap='Blues', display_labels=["Malignant", "Benign"])
plt.title("KNN classifier")
plt.grid(False)
plt.show()

# Plot the confusion matrix for the KNN classifier (with PCA)
plot_confusion_matrix(knn_pca, x_test_pca, y_test_pca, cmap='Blues', display_labels=["Malignant", "Benign"])
plt.title("KNN classifier(PCA)")
plt.grid(False)
plt.show()

# Plot the confusion matrix for the SVM classifier
plot_confusion_matrix(svm, x_test, y_test, cmap='Blues', display_labels=["Malignant", "Benign"])
plt.title("SVM classifier")
plt.grid(False)
plt.show()

"""
# Plot the confusion matrix for the SVM classifier
plot_confusion_matrix(svm_pca, x_test, y_test, cmap='Blues', display_labels=["Malignant", "Benign"])
plt.title("SVM classifier")
plt.grid(False)
plt.show()
"""
# ROC curve (for all the models)
roc_plot(x_test, y_test, x_test_pca, y_test_pca, [knn, svm], [knn_pca])







