import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

############ DATA PREPROCESSING 
# Load the Titanic dataset
titanic_data = pd.read_csv('D:\\internship\\titanic\\train.csv')

# # Drop the 'Cabin' column due to many missing values
# titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# # Fill missing values in the 'Age' column with the mean age
# titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

# # Fill missing values in the 'Embarked' column with the mode
# titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# # Convert categorical columns to numerical values
titanic_data = titanic_data.replace({'Sex': {'male': 0, 'female': 1},
                                      'Embarked': {'S': 0, 'C': 1, 'Q': 2}})

# # Explicitly cast the converted columns to integer type to avoid downcasting issues
# titanic_data['Sex'] = titanic_data['Sex'].astype(int)
# titanic_data['Embarked'] = titanic_data['Embarked'].astype(int)

# # Ensure there are no strings in the data
# print(titanic_data.head())

# Select the 5 features: Pclass, Sex, Age, SibSp, Parch
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
Y = titanic_data['Survived']


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# print(X.shape, X_train.shape, X_test.shape)

# # Initialize the Random Forest model
model = RandomForestClassifier(random_state=2)
model.fit(X_train, Y_train)

# Predict on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data:', training_data_accuracy)

# Predict on the testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data:', testing_data_accuracy)

# Print the number of features the model expects
print("Number of features expected by the model:", model.n_features_in_)

# Save the trained model to a file using pickle
filename = 'trained_random_forest_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)