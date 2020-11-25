
from PIL import Image

import SVC_optimized
from UserInput import *
from SVC import *
from preProcessing import *


st.write("""
# Cancer Diagnosis
Predict if someone is in the cancer risk group
""")
# Open and display image
image = Image.open('BreastCancer.jpg')
st.image(image, use_column_width=True)

# Get the data
#df = getData('/Users/johan.bsorensen/PycharmProjects/ML/data.csv')
df = getData('data.csv')

# Set target variable and map the target variable from string to numerical value
targetvar = 'diagnosis'

# Remove not useful columns
df[targetvar] = df[targetvar].map({'M':0, 'B':1})
df.drop(['id', 'Unnamed: 32'], inplace=True, axis=1)

# Manipulate the data set(Moving target column to be the last column)
cols = df.columns.tolist()
cols.remove(targetvar)
cols.append(targetvar)
df = df[cols]

# Feature selection
# Find and plot the optimal numbers of features, and select the most important ones
feat = rfe(df)
df = df[feat]

# Split the data into independent 'x' and dependent 'y' variables, and 80 % Training and 20% Testing
x_train, x_test, y_train, y_test = divideData(df)



# Create and train an optimal K - Nearest Neighbors Classifier
# svc = svc_classifier(x_train, y_train, x_test, y_test)
svc = SVC_optimized.svc_classifier(x_train, y_train, x_test, y_test)


user_input = getUserInput(df.columns)

# Runs when pushing then button
if st.sidebar.button('Run Classification'):
    # Store the model prediction in a variable
    svc_prediction = svc.predict(user_input)

    # Set a subheader and display the classification
    st.write("""
    # Classification
    """)
    # Show result of KNN
    if (svc_prediction == 1):
        st.write(' With a probability of ' + str(round(svc.predict_proba(user_input)[0][1] * 100, 2)) + '% Malignant.')
    else:
        st.write('With a probability of ' + str(round(svc.predict_proba(user_input)[0][0] * 100, 2)) + '% Benign.')

# Set a subheader
st.write("""
    # Information about the classifiers
    """)

st.header('Training data')
# Show the data as a table
st.dataframe(df)






