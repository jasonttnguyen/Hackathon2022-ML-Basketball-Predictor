from audioop import avg
import sklearn
import numpy
import pandas
import sklearn
from sklearn import linear_model
from datetime import timedelta



#This program predicts the stats of a given player with their csv file from basketball-reference.com (using linear regression algorithm)

playerData = pandas.read_csv("vanvleet2021.csv")

playerData = playerData[playerData["MP"].str.contains("Inactive") == False]
playerData = playerData[playerData["MP"].str.contains("Did Not Play") == False]
playerData = playerData[playerData["MP"].str.contains("Did Not Dress") == False]
playerData = playerData[playerData["FT%"].str.contains("NaN") == False]  #need to shift the games played/rnk down

playerData = playerData[["MP","FG","FGA","3P","3PA","FT","FTA","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS","+/-"]]

def seconder(x):  #get seconds for minutes played
	mins, secs = map(float, x.split(':'))
	td = timedelta(minutes=mins, seconds=secs)
	return td.total_seconds()
playerData['MP'] = playerData['MP'].apply(seconder)  #Take in mind, MP is not minutes played and actually seconds played


predictionVar = "PTS"

x = numpy.array(playerData.drop([predictionVar],1))  #creates an array of arrays containing the variables
y = numpy.array(playerData[predictionVar])           #creates an array of the wanted variable


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1 )  #trains model with 90 percent of data and test on 10% of data


linear = linear_model.LinearRegression()  #choosing linear regression as all stats are correalated 

linear.fit(x_train,y_train)  #fit model with the training data
accu = linear.score(x_test,y_test)  # used to print accuracy score
print("The accuracy is: "+ str(accu))

#["MP","FG","FGA","FG%","3P","3PA","3P%","FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS","+/-"]
avgMP=playerData["MP"].mean()
avgFG=playerData["FG"].mean()
avgFGA=playerData["FGA"].mean()
avg3P=playerData["3P"].mean()
avg3PA=playerData["3PA"].mean()
avgFT=playerData["FT"].mean()
avgFTA=playerData["FTA"].mean()
avgORB=playerData["ORB"].mean()
avgDRB=playerData["DRB"].mean()
avgTRB=playerData["TRB"].mean()
avgAST=playerData["AST"].mean()
avgSTL=playerData["STL"].mean()
avgBLK=playerData["BLK"].mean()
avgTOV=playerData["TOV"].mean()
avgPF=playerData["PF"].mean()
avgPTS=playerData["PTS"].mean()
avgPM=playerData["+/-"].sum()

averageStat = pandas.DataFrame({"MP":[avgMP],"FG":[avgFG],"FGA":[avgFGA],"3P":[avg3P],"3PA":[avg3PA],"FT":[avgFT],"FTA":[avgFTA],"ORB":[avgORB],"DRB":[avgDRB],"TRB":[avgTRB],"AST":[avgAST],"STL":[avgSTL],"BLK":[avgBLK],"TOV":[avgTOV],"PF":[avgPF],"PTS":[avgPTS],"+/-":[avgPM]})

print(averageStat)
#predictions = linear.predict(x_test)  #predict using x_test, which is randomized by sklearn
#for i in range(len(predictions)):
#    print(predictions[i],x_test[i],y_test[i])
