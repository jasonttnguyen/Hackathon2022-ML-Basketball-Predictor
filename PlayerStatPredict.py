import sklearn
import numpy
import pandas
import sklearn
from sklearn import linear_model
from datetime import timedelta

from sklearn.metrics import accuracy_score


#This program predicts the stats of the current star raptor players as of Oct 30, 2019 (using linear regression algorithm)
#This formats all the data so the most recent game is on the right, and last played game is on the left, as well as formatting the data given so it is easier to work with
playerData = pandas.read_csv("vanvleet2021.csv")

playerData = playerData[playerData["MP"].str.contains("Inactive") == False]
playerData = playerData[playerData["MP"].str.contains("Did Not Play") == False]
playerData = playerData[playerData["MP"].str.contains("Did Not Dress") == False]
playerData = playerData[playerData["FT%"].str.contains("NaN") == False]  #need to shift the games played/rnk down

playerData = playerData[["MP","FG","FGA","FG%","3P","3PA","3P%","FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS","+/-"]]

def seconder(x):  #get seconds for minutes played
	mins, secs = map(float, x.split(':'))
	td = timedelta(minutes=mins, seconds=secs)
	return td.total_seconds()
playerData['MP'] = playerData['MP'].apply(seconder)



predictionVar = "PTS"

print("DEBUG HERE") #Debug###########################################
x = numpy.array(playerData.drop([predictionVar],1))  #creates an array of arrays containing the variables
y = numpy.array(playerData[predictionVar])           #creates an array of the wanted variable
print("This is x: \n") #Debug###########################################
print(x)               #Debug###########################################

print("This is y: \n")  #Debug###########################################
print(y)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1 )  #trains model with 90 percent of data and test on 10% of data


linear = linear_model.LinearRegression()  #choosing linear regression as all stats are correalated 

linear.fit(x_train,y_train)  #fit model with the training data
accu = linear.score(x_test,y_test)  # used to print accuracy score
print("The accuracy is: "+ str(accu))

predictions = linear.predict(x_test)  #predict using x_test, which is randomized by sklearn
print(len(predictions)) #Debug###########################################
for i in range(len(predictions)):
    print(predictions[i],x_test[i],y_test[i])
