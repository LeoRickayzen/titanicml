import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.metrics import f1_score
import re

def ticketClass(row):
	if row['Ticket'] != 'LINE':
		m = re.search('(?:.* )?([0-9]*)', row['Ticket'])
		#print(row['Ticket'] + ": " + m.group(1)[0])
		if m.group(1)[0] == 0:
			return 0
		else:
			return m.group(1)[0]
	else:
		return 0

def crew(row):
    if row['Fare'] == 0:
        return 1
    else:
        return 0

def getLabels(filename):
	df = pd.read_csv(filename)
	labels = df['Survived']
	return labels

def getFeatures(filename):
	df = pd.read_csv(filename)

	df_copy = df.copy()

	df_copy['Cabin'].replace({r'^.*[A-Z]([0-9]*).*$' : r'\1'}, regex=True, inplace=True)

	df['Embarked'].replace('Q', 0, inplace=True)
	df['Embarked'].replace('S',0.5 ,inplace=True)
	df['Embarked'].replace('C',1 ,inplace=True)

	df['Sex'].replace('male', 0, inplace=True)
	df['Sex'].replace('female',1 ,inplace=True)

	df['Cabin'].replace('A.*', 0.1, regex=True, inplace=True)
	df['Cabin'].replace('B.*', 0.2, regex=True, inplace=True)
	df['Cabin'].replace('C.*', 0.3, regex=True, inplace=True)
	df['Cabin'].replace('D.*', 0.4, regex=True, inplace=True)
	df['Cabin'].replace('E.*', 0.5, regex=True, inplace=True)
	df['Cabin'].replace('F.*', 0.6, regex=True, inplace=True)
	df['Cabin'].replace('G.*', 0.7, regex=True, inplace=True)
	df['Cabin'].replace('T', np.NaN, regex=True, inplace=True)

	df.rename(index=str, columns={"Cabin": "Deck"}, inplace=True)

	df['Cabin'] = df_copy['Cabin'].values

	df['FamilySize'] = df['Parch'] + df['SibSp']
	df['TicketClass'] = df.apply(ticketClass, axis=1)
	df['Crew'] = df.apply(crew, axis=1)

	df['Cabin'] = df['Cabin'].apply(pd.to_numeric)
	df['Deck'] = df['Deck'].apply(pd.to_numeric)
	df['TicketClass'] = df['TicketClass'].apply(pd.to_numeric)
	#features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
	#features = df[['Pclass', 'Age', 'Parch', 'SibSp', 'Fare']]
	#features = df[['Age', 'Sex', 'Parch', 'Fare', 'Pclass', 'SibSp', 'Embarked', 'Deck', 'TicketClass']]
	features = df[['Age', 'Sex', 'Parch', 'Fare', 'Pclass', 'SibSp', 'Embarked', 'Deck', 'TicketClass', 'Crew']]

	return df

def sanitizeFeatures(features):
	min_max_scaler = preprocessing.MinMaxScaler()

	features_arr = features.values

	#next 3 lines (81-87)
	#author: 'Daniel'
	#date: September 8th 2013
	#url: https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
	col_mean = np.nanmean(features_arr,axis=0)

	#Find indicies that you need to replace
	inds = np.where(np.isnan(features_arr))

	#Place column means in the indices. Align the arrays using take
	features_arr[inds]=np.take(col_mean,inds[1])

	features_arr = min_max_scaler.fit_transform(features_arr)

	return features_arr

features = getFeatures('train.csv')
labels = getLabels('train.csv')
test_features = getFeatures('test.csv')

d = {'features_config': ['A', 'B', 'C', 'D', 'E', 'F'], 
	'Structure': ['10', '22, 12', '22, 5', '18, 11', '18, 11', '18, 11']
	}

networks = [MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10), random_state=1, early_stopping=True),
				MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(22, 12), random_state=1, early_stopping=True),
				MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(22, 5), random_state=1, early_stopping=True),
				MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18, 11), random_state=1, early_stopping=True),
				MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18, 11), random_state=1, early_stopping=True),
				MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18, 11), random_state=1, early_stopping=True)]

svms = [svm.SVC(C=150000),
		svm.SVC(C=150000),
		svm.SVC(C=150000),
		svm.SVC(C=150000),
		svm.SVC(C=100000),
		svm.SVC(C=100000)]

rfs = [RandomForestClassifier(max_depth=6, random_state=0, n_estimators=2),
		RandomForestClassifier(max_depth=6, random_state=0, n_estimators=2),
		RandomForestClassifier(max_depth=6, random_state=0, n_estimators=5),
		RandomForestClassifier(max_depth=6, random_state=0, n_estimators=2),
		RandomForestClassifier(max_depth=6, random_state=0, n_estimators=2),
		RandomForestClassifier(max_depth=6, random_state=0, n_estimators=4)]

results = pd.DataFrame(data=d)

mlp_valid_scores = np.array([])
mlp_train_scores = np.array([])
mlp_f_scores = np.array([])

svm_valid_scores = np.array([])
svm_train_scores = np.array([])
svm_f_scores = np.array([])

rf_valid_scores = np.array([])
rf_train_scores = np.array([])
rf_f_scores = np.array([])

features_sets = [features[['Age', 'Sex']], 
				features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare']], 
				features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked']], 
				features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Deck']], 
				features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Deck', 'TicketClass']], 
				features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Deck', 'TicketClass', 'Crew']]]

test_features_sets = [test_features[['Age', 'Sex']], 
					test_features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare']], 
					test_features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked']], 
					test_features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Deck']], 
					test_features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Deck', 'TicketClass']], 
					test_features[['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare', 'Embarked', 'Deck', 'TicketClass', 'Crew']]]

for i in range(0, len(features_sets)):

	features_arr = sanitizeFeatures(features_sets[i])
	labels_arr = labels.values
	test_features_arr = sanitizeFeatures(test_features_sets[i])

	labels_train = labels_arr[:800]
	features_train = features_arr[:800]
	
	labels_validate = labels_arr[800:]
	features_validate = features_arr[800:]
	
	clfRf = rfs[i]
	clfSvm = svms[i]
	clfMlp = networks[i]
	
	clfSvm.fit(features_train, labels_train)
	clfMlp.fit(features_train, labels_train)
	clfRf.fit(features_train, labels_train)
	
	d = {'PassengerId' : pd.read_csv('test.csv')['PassengerId'], 'Survived' : clfMlp.predict(test_features_arr)}
	predictions = pd.DataFrame(data = d)
	predictions.to_csv('mlppredictions/predictedmlp' + str(i) + '.csv' , index=False)

	d = {'PassengerId' : pd.read_csv('test.csv')['PassengerId'], 'Survived' : clfSvm.predict(test_features_arr)}
	predictions = pd.DataFrame(data = d)
	predictions.to_csv('svmpredictions/predictedsvm' + str(i) + '.csv' , index=False)

	d = {'PassengerId' : pd.read_csv('test.csv')['PassengerId'], 'Survived' : clfRf.predict(test_features_arr)}
	predictions = pd.DataFrame(data = d)
	predictions.to_csv('rfpredictions/predictedrf' + str(i) + '.csv' , index=False)
	
	svm_train = round(clfSvm.score(features_train, labels_train), 3)
	svm_test = round(clfSvm.score(features_validate, labels_validate), 3)
	svm_f1 = round(f1_score(clfSvm.predict(features_validate), labels_validate, average='binary'), 3)

	mlp_train = round(clfMlp.score(features_train, labels_train), 3)
	mlp_test = round(clfMlp.score(features_validate, labels_validate), 3)
	mlp_f1 = round(f1_score(clfMlp.predict(features_validate), labels_validate, average='binary'), 3)

	rf_train = round(clfRf.score(features_train, labels_train), 3)
	rf_test = round(clfRf.score(features_validate, labels_validate), 3)
	rf_f1 = round(f1_score(clfRf.predict(features_validate), labels_validate, average='binary'), 3)

	mlp_valid_scores = np.append(mlp_valid_scores, mlp_test)
	mlp_train_scores = np.append(mlp_train_scores, mlp_train)
	mlp_f_scores = np.append(mlp_f_scores, mlp_f1)
	
	svm_valid_scores = np.append(svm_valid_scores, svm_test)
	svm_train_scores = np.append(svm_train_scores, svm_train)
	svm_f_scores = np.append(svm_f_scores, svm_f1)
	
	rf_valid_scores = np.append(rf_valid_scores, rf_test)
	rf_train_scores = np.append(rf_train_scores, rf_train)
	rf_f_scores = np.append(rf_f_scores, svm_f1)

	print("svm train: " + str(clfSvm.score(features_train, labels_train)))
	print("svm test: " + str(clfSvm.score(features_validate, labels_validate)))
	print("svm test f1: " + str(f1_score(clfSvm.predict(features_validate), labels_validate, average='binary')))
	
	print("mlp train: " + str(clfMlp.score(features_train, labels_train)))
	print("mlp test: " + str(clfMlp.score(features_validate, labels_validate)))
	print("mlp test f1: " + str(f1_score(clfMlp.predict(features_validate), labels_validate, average='binary')))
	
	print("rf train: " + str(clfRf.score(features_train, labels_train)))
	print("rf test: " + str(clfRf.score(features_validate, labels_validate)))
	print("rf test f1: " + str(f1_score(clfRf.predict(features_validate), labels_validate, average='binary')))

results['mlp_validation'] = mlp_valid_scores
results['mlp_train'] = mlp_train_scores
results['mlp_f_scores'] = mlp_f_scores

results['svm_validation'] = svm_valid_scores
results['svm_train'] = svm_train_scores
results['svm_f_scores'] = svm_f_scores

results['rf_validation'] = rf_valid_scores
results['rf_train'] = rf_train_scores
results['rf_f_scores'] = rf_f_scores

results.to_csv('results.csv', index=False)