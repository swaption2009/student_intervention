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

