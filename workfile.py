# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"


# EXPLORE DATA

# TODO: Calculate number of students
n_students = np.shape(student_data)[0]

# TODO: Calculate number of features
n_features = np.shape(student_data)[1]

# TODO: Calculate passing students
students_who_pass = np.where(student_data["passed"] == "yes")
n_passed = np.size(students_who_pass)

# TODO: Calculate failing students
students_who_fail = np.where(student_data["passed"] == "no")
n_failed = np.size(students_who_fail)

# TODO: Calculate graduation rate
grad_rate = float(n_passed) / float(n_students) * 100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# PREPARE DATA

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


# PREPROCESSING (using Panda get_dummies)

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

# Train/test data split

# TODO: Import any additional functionality you may need here
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42)

# TODO: Set the number of training points
num_train = X_train.shape[0]

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train = X_train
X_test = X_test
y_train = y_train
y_test = y_test

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# SETUP

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))



# 3 classifiers: logistics regression, naive bayes, and decision tree

# TODO: Import the three supervised learning models from sklearn
# from sklearn import model_A
from sklearn import linear_model
# from sklearn import model_B
from sklearn.naive_bayes import MultinomialNB
# from skearln import model_C
from sklearn.tree import DecisionTreeClassifier

# TODO: Initialize the three models
clf_A = linear_model.LogisticRegression()
clf_B = MultinomialNB()
clf_C = DecisionTreeClassifier(random_state=0)

# TODO: Set up the training set sizes
X_train_100, X_test_100, y_train_100, y_test_100 = train_test_split(X_all, y_all, test_size=0.75, random_state=42)
X_train_100 = X_train_100
y_train_100 = y_train_100

X_train_200, X_test_200, y_train_200, y_test_200 = train_test_split(X_all, y_all, test_size=0.5, random_state=42)
X_train_200 = X_train_200
y_train_200 = y_train_200

X_train_300, X_test_300, y_train_300, y_test_300 = train_test_split(X_all, y_all, test_size=0.25, random_state=42)
X_train_300 = X_train_300
y_train_300 = y_train_300

# TODO: Execute the 'train_predict' function for each classifier and each training set size

# set test_size of 0.25 (296 training data)
train_predict(clf_A, X_train_100, y_train_100, X_test_100, y_test_100)
train_predict(clf_B, X_train_100, y_train_100, X_test_100, y_test_100)
train_predict(clf_C, X_train_100, y_train_100, X_test_100, y_test_100)

# set test_size of 0.5 (296 training data)
train_predict(clf_A, X_train_200, y_train_200, X_test_200, y_test_200)
train_predict(clf_B, X_train_200, y_train_200, X_test_200, y_test_200)
train_predict(clf_C, X_train_200, y_train_200, X_test_200, y_test_200)

# set test_size of 0.75 (296 training data)
train_predict(clf_A, X_train_300, y_train_300, X_test_300, y_test_300)
train_predict(clf_B, X_train_300, y_train_300, X_test_300, y_test_300)
train_predict(clf_C, X_train_300, y_train_300, X_test_300, y_test_300)

