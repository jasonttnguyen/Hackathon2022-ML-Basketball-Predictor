from audioop import avg
from click import prompt
import sklearn
import numpy
import pandas
import sklearn
from sklearn import linear_model
from datetime import timedelta



#This program predicts the stats of a given player with their csv file from basketball-reference.com (using linear regression algorithm)
userPlayerChoice=input("\nWhich Toronto Raptor starter would you like to use? \nEnter:\n(1) for Fred VanVleet\n(2) for Pascal Siakam\n(3) for Gary Trent Jr.\n(4) for Scottie Barnes\n(5) for OG Anunoby: \n")
print("\n")
playerName = "NULL"
if userPlayerChoice=="1":
	playerName = "Fred Vanvleet"
	choice="vanvleet2021.csv"

elif userPlayerChoice=="2":
	playerName = "Pascal Siakam"
	choice="siakam2021.csv"

elif userPlayerChoice=="3":
	playerName = "Gary Trent Jr"
	choice="trentjr2021.csv"

elif userPlayerChoice=="4":
	playerName = "Scottie Barnes"
	choice="barnes2021.csv"

elif userPlayerChoice=="5":
	playerName = "OG Anunoby"
	choice="anunoby2021.csv"

else:
	print("Invalid choice: Vanvleet's data will be used as default")
	playerName = "Vanvleet"
	choice="vanvleet2021.csv"

print("\nChosen player: " + playerName)

playerData = pandas.read_csv(choice) #insert player csv file here, make sure path is correct. 

playerData = playerData[playerData["MP"].str.contains("Inactive") == False]
playerData = playerData[playerData["MP"].str.contains("Did Not Play") == False]
playerData = playerData[playerData["MP"].str.contains("Did Not Dress") == False]
playerData = playerData[playerData["FT%"].str.contains("NaN") == False]  #need to shift the games played/rnk down

playerData = playerData[["MP","FGA","TRB","AST","STL","BLK","TOV","PTS","+/-"]]

def seconder(x):  #get seconds for minutes played
	mins, secs = map(float, x.split(':'))
	td = timedelta(minutes=mins, seconds=secs)
	return td.total_seconds()
playerData['MP'] = playerData['MP'].apply(seconder)  #Take in mind, MP is not minutes played and actually seconds played


predictionVar = "PTS"

x = numpy.array(playerData.drop([predictionVar],axis=1))  #creates an array of arrays containing the variables
y = numpy.array(playerData[predictionVar])           #creates an array of the wanted variable


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.05 )  #trains model with 95 percent of data and test on 5% of data


linear = linear_model.LinearRegression()  #choosing linear regression as all stats are correalated 

linear.fit(x_train,y_train)  #fit model with the training data



predictions = linear.predict(x_test)  #predict using x_test, which is randomized by sklearn
print("\nGiven that in a certain randomized game, " + playerName + " achieves the following stats: ")
print("Seconds Played: " + str(x_test[1][0]) + 
	"\nField Goals Attempted: "+ str(x_test[1][1]) +  
	"\nRebounds: " + str(x_test[1][2]) +
	"\nAssists: " + str(x_test[1][3]) + 
	"\nSteals: " + str(x_test[1][4]) + 
	"\nBlocks:" + str(x_test[1][5]) + 
	"\nTurnovers: " + str(x_test[1][6]) +
	"\n+/-: " + str(x_test[1][7]) )
userGuess=input("\nWhat is your prediction on "+ playerName +" point production that game?: ")
print("The machine learning model predicted: " + str(predictions[1]))
print("The actual point score is: " + y_test[1]) 
userGuessAcc = int(userGuess)/int(y_test[1])
mlAcc=int(predictions[1])/int(y_test[1])
print("\nThe accuracy of your guess is: " + str(userGuessAcc))
print("The accuracy of the machine learning model is: " + str(mlAcc))

if userGuessAcc>mlAcc:
	print("\n2Congratulations! You are more accurate than the machine learning model!")

#for i in range(len(predictions)):   #uncomment out if you want to see the test data and the predictions for thos
#  print("The predicted output generated is: " + str(predictions[i]) + "The test data used to generate the prediction was: " + str(x_test[i]) + "The actual output was: " + str(y_test[i]))
