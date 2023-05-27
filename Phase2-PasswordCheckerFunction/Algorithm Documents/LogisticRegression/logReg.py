import numpy as np # is the core library for scientific computing in Python. It provides a high-performance multidimensional array object and tools for working with these arrays.
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas # for data processing from the .csv file that has the dataset
from sklearn.model_selection import train_test_split # cross validation class
from sklearn.linear_model import LogisticRegression # Logistic Regression Algorithm's class

# a function for vectorization (NLP - Natural Language Processing)
# A token that can refer to words, subwords, or even characters. 
# Tokenization basically refers to the process of seperating a whole into its constituent parts.
def getTokens(inputString): # custom tokenizer. ours tokens are characters rather than full words
    tokens = [*inputString]
    return tokens
# The purpose of the above fuction (the purpose of using a tokenizer) is to split a password which is a string into its characters which are a constituent part, and these constituent parts are named 'token'
# Or it can be as follows:

''' or
def getTokens(inputString): #custom tokenizer. ours tokens are characters rather than full words
	tokens = []
	for i in inputString:
		tokens.append(i)
	return tokens
'''

def AI_Logistic_Regression():
    # Step1: Data Fetching
    filepath = 'c:/Users/fufu_/Desktop/GraduationProject/ProjectWorkshop/Phase2-PasswordCheckerFunction/withLogisticRegression/data.csv' # path for password file
    data = pandas.read_csv(filepath,',',error_bad_lines=False)
    data = pandas.DataFrame(data).values.astype('U')
    passwords = np.array(data) # here, converting the dataframe to a two-dimesional in other words, rank 2 array

    # Step2: Data Preprocessing
    random.shuffle(passwords) #shuffling randomly for robustness

    # splitting the 'passwords' rank 2 array to two as 'y' which represents the target (dependent) column of the passwords rank 2 array and as 'allpasswords' which represents the independent (feature) column of the passwords rank 2 array 
    y = [d[1] for d in passwords] # labels (column2 of the passwords rank 2 array)
    allpasswords= [d[0] for d in passwords] # actual passwords (column1 of the passwords rank 2 array)

    vectorizer = TfidfVectorizer(tokenizer=getTokens) # vectorization that is the process of transforming a token to a vector as linear algebra to prepare the data for logistic regression
    X = vectorizer.fit_transform(allpasswords) # transfroming the tokens to vectors
    
    # Step3: Splitting Train dataset and test dataset through cross validation 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Step4: Classification
    classifier = LogisticRegression(penalty='l2',multi_class='ovr')  # our logistic regression classifier
    classifier.fit(X_train, y_train) # training
    score = classifier.score(X_test, y_test)  # testing

    # Step5: Prediction
    X_predict = ['A+li','faizanahmad123','faizanahmad##','ajd1348#28t**','ffffffffff','kuiqwasdi','uiquiuiiuiuiuiuiuiuiuiuiui','mynameisfaizan','mynameis123faizan#','faizan','123456','abcdef']
    X_predict = vectorizer.transform(X_predict)
    y_predict = classifier.predict(X_predict)
    return y_predict

print(AI_Logistic_Regression())

# NOTES: 
# Why do we vectorize text data?
# In order to perform machine learning on text, we need to transform our documents into vector representations such that we can apply numeric machine learning. 
# This process is called feature extraction or more simply, vectorization, and is an essential first step toward language-aware analysis.
# Vectorization is used to speed up the Python code without using loop. 
# Using such a function can help in minimizing the running time of code efficiently.
