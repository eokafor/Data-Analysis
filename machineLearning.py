import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy import stats
import seaborn


OUTPUT_TEMPLATE = (
    'KNeighborsClassifier valid score: {score:.3g}\n'
    'mannwhitneyu p-value: {test:.3g}\n'
)

def main():
	seaborn.set()

	#read the .csv file and create a datframe
	data = pd.read_csv('subject_walk_pace_results.csv', sep=',')

	#split the data frame ito 2, to have a second data set to predict
	df1 = data.iloc[:22,:]
	df2 = data.iloc[22:,:]
	df3 = df2.drop(columns=['gender'])

	#generate a column of unknown genders to later be predicted by the chosen classifier
	gender = ['unknown', 'unknown', 'unknown', 'unknown','unknown','unknown','unknown','unknown','unknown']
	df3['gender'] = gender

	#extract the walking pace from dataframe and reshape to a 2D array
	X = df1['walking_pace'].values
	X = X.reshape(-1,1)
	y = df1['gender'].values

	#train and test the known input and outputs of the first half split dataframe (df1)
	X_train, X_valid, y_train, y_valid = train_test_split(X, y)


	#perform a machine learning classification of the trained data
	#model = GaussianNB()
	model = KNeighborsClassifier(n_neighbors=5)
	#model = DecisionTreeClassifier(max_depth=5, random_state=42)

	#fit the data to the model
	model.fit(X_train, y_train)

	#print(model.score(X_train, y_train))


	#using the second half of the split dataframe, predit the output genders
	X_unknown = df3['walking_pace'].values
	X_unknown = X_unknown.reshape(-1,1)
	predictions = model.predict(X_unknown)
	#write the result into a csv file provided in the command line
	#pd.Series(predictions).to_csv(sys.argv[1], index=False, header=False)
	pd.Series(predictions).to_csv('predictions.csv', index=False, header=False)


	#Statistical Analysis
	#copy the original dataframe to avaid unnecessary changes
	stat_data = data.copy()

	#divive the dataframe into 2 different groups, male and female
	male_data = stat_data[stat_data["gender"] == "Male"]
	female_data = stat_data[stat_data["gender"] == "Female"]

	xa = male_data["walking_pace"]
	xb = female_data["walking_pace"]

	#draw a histogram to better visualise the data
	plt.hist(xa, label='Male')
	plt.hist(xb, label='Female')
	plt.legend(loc="upper right")
	plt.title('Walking Pace distribution amongst genders')
	plt.xlabel('walking pace (steps/secs)')
	plt.ylabel('no_occurrence')
	#plt.savefig(sys.argv[2])
	plt.savefig('hist_plot.png')
	plt.close()
	#plt.show()

	#print(stats.normaltest(xa).pvalue)
	#print(stats.normaltest(xb).pvalue)

	#print(stats.levene(xa, xb).pvalue)

	#perform a statisical test and print out the p-vale as well as the model valid score
	print(OUTPUT_TEMPLATE.format(
		score = model.score(X_valid, y_valid),
		test = stats.mannwhitneyu(xa, xb).pvalue,
	))

	#ttest = stats.ttest_ind(xa, xb)
	#print(ttest.pvalue)

	

if __name__ == '__main__':
    main()