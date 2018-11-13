# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

"""
#Deprecated 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

# create dummy variable as we have more than 2 category, to avoid dummy variable trap
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# we need to do a onehot encoding of country and lable encoding of gender.
# actually we dont need to pipeline, because labelencoder cannot be pipelined with onehotencoder
# as labelencoder returns y like array whereas onehotencoder returns X like array. so if we pipeline
# tuple

"""
country = ('country_onehot', OneHotEncoder())
department = ('department', OneHotEncoder())

# list of encoding steps
encoder = [country,department]
# columns to be encoded in pipeline
columns = [1,2]

# define pipeline of the encoder steps
pipeline = Pipeline(encoder)

# define the columntransformation

transformers = [('country&department',pipeline,columns)]
columntransformer = ColumnTransformer(transformers=transformers)

X = columntransformer.fit_transform(X)
"""

labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

transformer = [('country', OneHotEncoder(), [1])]
columntransformer = ColumnTransformer(transformers=transformer, remainder='passthrough')  # default is drop
X = columntransformer.fit_transform(X)

# notice onehotencoding will push the onehotcolumns to the beginning of the array automatically while fitting
# this helps in removing one of the dummy variables from the X

X = X[:, 1:]  # removed the first dummy variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# import the keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense, Dropout

"""
# Create your classifier here
# Initializing the ANN
classifier = Sequential()

# select average of input & output nodes as number of hidden layers
# need to specify inputshape only at the beginning

# 1st hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11, )))

# Dropout for regularization
classifier.add(Dropout(rate=0.1))

# 2nd hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))

# Dropout for regularization
classifier.add(Dropout(rate=0.1))

# Output Layer
# in multiclass classification use activation = 'softmax' and unit as number of output classes
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

# for multiclass classfication use 'categorical_crossentropy'
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting classifier to the Training set

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred > 0.5)

"""
"""
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40 years old
    Tenure: 3 years
    Balance: $60000
    Number of Products: 2
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $50000
"""
# new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) > 0.5


"""
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def buildclassifier():
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=buildclassifier, batch_size=10, epochs=100)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
mean = accuracies.mean()
std = accuracies.std()

"""

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def buildclassifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=buildclassifier)

parameters = {'batch_size': [25, 32],
              'epochs': [500, 100],
              'optimizer': ['rmsprop', 'adam']}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X=X_train, y=y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("The best parameters identified by grid search are: "+ str(best_parameters))
print("the best accuracy obtained by above parameters is: "+ str(best_accuracy))