# PROBLEM SCENARIO :-
# As a HR manager working for an employer of 10 year old and wants to hire experienced cnadidate. Since they are only 10 year old
# they only have the data fom last 10 years for salary provided to their employees. HR wants to know what salary to be provided for a 15 year experience candidate 
# using AI model

# The pandas library is imported when we try to use a datatable/database.
import pandas as pd
import pickle

# using sklearn library for splitting training and test set.
from sklearn.model_selection import train_test_split
# importing linear regresion and r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# using read.csv function from pandas library to read the dataset
dataset = pd.read_csv('SalaryData.csv')
print(dataset)

# storing only YearsExperience column details to a variable
independent = dataset[["YearsExperience"]]
print(independent)
# storing only Salary column details to a variable
dependent = dataset[["Salary"]]
print(dependent)

# splitting the test and train data and applying test size of 30% by providing independent an dependent variable as a parameter to the "train_test_split" function
x_train,x_test,y_train,y_test = train_test_split(independent,dependent, test_size = 0.30, random_state = 0)
# train_test_split function will give four outputs so thats why we are storing those four output to four variables (x_train,x_test,y_train,y_test) and we can use that as an
# input to another function.
print(x_train,x_test,y_train,y_test)

# storing linearregression class to a variable, so that it stores all the LineraRegression function that variable 
regressor = LinearRegression()

#  calling "fit" function from "LineraRegression" library/class and including respective input and output training sets as parameter.
regressor.fit(x_train,y_train) 

# calculating weight and bias i.e the formula Y(output)=w(weight)X + b(bias) ,"weight" means distance between two points/values and "bias" means origin/starting point
weight = regressor.coef_
print(weight)
bias = regressor.intercept_
print(bias)

# now we are cross checking to evaluating it by  providing test data "x_test" as an input parameter
y_pred = regressor.predict(x_test)

# predicting r_score which provides prediction value from metrics function
r_score = r2_score(y_test,y_pred)  
print(r_score)

# saving the above model to a fle using "pickle" function and predicting the value/salary through the saved model
fileName = "finalised_model_linear.sav"
pickle.dump(regressor,open(fileName,'wb'))
loaded_model = pickle.load(open('finalised_model_linear.sav','rb'))
result = loaded_model.predict([[13]])
print(result)
