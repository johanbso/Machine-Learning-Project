import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

"""
Transforms csv-file to a pandas dataFrame
@:param fileName - Name of csv file
@:return dataFrame - Data frame containing the training/testing data
"""
def getData(fileName):
    # importing the dataset
    dataFrame = pd.read_csv(fileName)
    return dataFrame

def rfe(df):
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit(x, y)

    #print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    """
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    """


    features = [f for f, s in zip(x.columns, rfecv.support_) if s]
    features.append('diagnosis')
    return features

"""
Plots the features importance using a number of randomized decision trees 
@:param dataFrame - dataFrame containing the training/testing data
"""
def featureImportance(dataFrame):
    plt.rcParams['figure.figsize'] = 15, 6
    sns.set_style("darkgrid")

    x = dataFrame.iloc[:, :-1]
    y = dataFrame.iloc[:, -1]

    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    feat_importances.sort_values().plot(kind='barh')
    plt.show()
    return feat_importances.sort_values().index

"""
Removes all features not selected
@:param dataFrame - Data frame containing the training/testing data
@:param dropColumns - Vector containing the name of the columns to be removed from the dataset
@:return df - Data frame with only the selecte features and the target values
"""
def featureReduction(dataFrame, dropColumns):

    df = dataFrame.drop(dropColumns, axis=1)

    return df

"""
Split the data in to 4 vectors, one containing the training features(x_train), one containing the training target values
(y_train), one containing the test features(x_test), and one containing the test target values(y_test). 
@:param df - Data frame containing the training/testing data
@:return x_train, x_test, y_train, y_test - Vectors containing the data used in training and testing
"""
def divideData(df):
    x = df.iloc[:, :(len(df.columns) - 1)].values
    y = df.iloc[:, -1].values
    # random state = ?
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

"""
Standardize the features
@:param x_train - Training feature vector
@:param x_test - Test feature vector
@:return x_train, x_test - Vectors containing scaled data
"""
def featureScaler(x_train, x_test):

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test

"""
Plots features in pairs, in order to analyze correlation.
@:param dataFrame - Data frame containing traning data
@:param targetColumn - Name of the column containing the target values
"""
def runPairPlot(df):
    targetvalue = "Diagnosis"
    df = df.rename(columns={'diagnosis': targetvalue}, inplace=False)
    df[targetvalue] = df[targetvalue].map({0: 'Malignant', 1: 'Benign'})
    sns.pairplot(df, hue=targetvalue)
    plt.show()


"""
Perform Principal Component Analysis, reducing the dimensions of the data set.
@:param df - DataFrame containing training and test data
@:targetvar - The target variable
@:return - Transformed DataFrame with fewer dimensions
"""
def pca(df, targetvar):
    features = []
    for feature in df:
        if feature != targetvar:
            features.append(feature)
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,[targetvar]].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pcaa = PCA(n_components=2)
    principalComponents = pcaa.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[[targetvar]]], axis=1)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[targetvar] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(['Diagnosis = Malignant ', 'Diagnosis = Benign'])
    ax.grid()
    plt.show()
    """
    return finalDf


"""
Create boxplot, used in outlier detection.
@:param df - DataFrame containing training and test data
@:targetvar - The target variable
"""
def boxplot(df):
    targetvalue = "Diagnosis"
    df = df.rename(columns={'diagnosis': targetvalue}, inplace=False)
    df[targetvalue] = df[targetvalue].map({0: 'Malignant', 1: 'Benign'})

    y = df[targetvalue] # M or B
    x = df.drop(targetvalue, axis=1)
    ax = sns.countplot(y, label="Count")  # M = 212, B = 357
    plt.show()

    data = x
    data_n_2 = (data - data.mean()) / (data.std())  # standardization
    data = pd.concat([y, data_n_2.iloc[:, 0:8]], axis=1)
    data = pd.melt(data, id_vars=targetvalue,
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.boxplot(x="features", y="value", hue=targetvalue, data=data)
    plt.xticks(rotation=90)
    plt.show()

    data = pd.concat([y, data_n_2.iloc[:, 8:]], axis=1)
    data = pd.melt(data, id_vars=targetvalue,
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.boxplot(x="features", y="value", hue=targetvalue, data=data)
    plt.xticks(rotation=90)
    plt.show()

"""
Create violinplot (histograms), used in feature selection .
@:param df - DataFrame containing the data set
"""
def violin_plot(dataFrame):
    targetvalue = "Diagnosis"
    df = dataFrame.rename(columns={'diagnosis': targetvalue}, inplace=False)
    df[targetvalue] = df[targetvalue].map({0:'Malignant',  1:'Benign'})

    data = df.drop([targetvalue], axis = 1)
    data_n_2 = (data - data.mean()) / (data.std())  # standardization

    # Plot histograms for the 10 first features
    data = pd.concat([df[targetvalue], data_n_2.iloc[:, 0:10]], axis=1)
    data = pd.melt(data, id_vars=targetvalue,
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue=targetvalue, data=data, split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.show()

    # Plot histograms for the 11-20 first features
    data = pd.concat([df[targetvalue], data_n_2.iloc[:, 10:20]], axis=1)
    data = pd.melt(data, id_vars=targetvalue,
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue=targetvalue, data=data, split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.show()

    # Plot histograms for the 21-> first features
    data = pd.concat([df[targetvalue], data_n_2.iloc[:, 20:]], axis=1)
    data = pd.melt(data, id_vars=targetvalue,
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue=targetvalue, data=data, split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.show()






