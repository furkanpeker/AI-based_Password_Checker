''' General and Common Imports '''
# Library for fetching the credentials
import paramiko
# Libraries for AI
import numpy as np # is the core library for scientific computing in Python. It provides a high-performance multidimensional array object and tools for working with these arrays.
import random
import pandas # for data processing from the .csv file that has the dataset

''' For Decision Tree Algorithm'''
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from array import *
from sklearn.tree import DecisionTreeClassifier

''' For Logistic Regression Algorithm'''
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split # cross validation class
from sklearn.linear_model import LogisticRegression # Logistic Regression Algorithm's class

''' Global declerations for fetchCredentials() and detectLogicalLocationOfWeaks() functions '''
hostname = input('Hostname: ')
usernameOfPfsense = input('Username: ')
passwordOfPfsense = input('password: ')

''' General Auxilary Functions '''
def fetchCredentials():
    # copy the file which has the users' identifies from the RADIUS database:
    # Getting the identifies
    ssh_session = paramiko.SSHClient()
    port = int(input('Port: '))
    # Establishing the connection
    ssh_session.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_session.connect(hostname = hostname, username = username, password = password, port = port)
    # Implementation phase
    sftp_client = ssh_session.open_sftp()
    # fetching the file from the server:
    sftp_client.get('/usr/local/etc/raddb/users', 'users.txt')
    ssh_session.close()

def getPasswords():
    # open the users.txt, fetch the data / content and converting it to a list " "
    f = open('users.txt', 'r')
    data = f.read()
    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.replace('\n', ' ').split(" ")
    f.close()

    passwords = []

    passwords.append(data_into_list[4].strip('"')) # extracting passwords from data_into_list into the passwords list by getting rid of the (")s 
    i=14
    while (i < len(data_into_list)):
        passwords.append(data_into_list[i].strip('"'))
        i+=10
    #print(passwords)
    return passwords

''' Auxilary Function for the Logistic Regression Algrorithm: '''
# a function for vectorization (NLP - Natural Language Processing)
# A token that can refer to words, subwords, or even characters. 
# Tokenization basically refers to the process of seperating a whole into its constituent parts.
def getTokens(inputString): # custom tokenizer. ours tokens are characters rather than full words
    tokens = [*inputString]
    return tokens
# The purpose of the above fuction (the purpose of using a tokenizer) is to split a password which is a string into its characters which are a constituent part, and these constituent parts are named 'token'
# Or it can be as follows:

''' Auxilary Functions for the Decision Tree Algrorithm: '''
def getName_Surnames():
    # open the users.txt, fetch the data / content and converting it to a list " "
    f = open('users.txt', 'r')
    data = f.read()
    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.replace('\n', ' ').split(" ")
    f.close()

    names = []
    surnames = []

    # extracting names from data_into_list into the nameSurname list by getting rid of the (")s 
    names.append(data_into_list[5].strip('"')) 
    i=15
    while (i < len(data_into_list)):
        names.append(data_into_list[i].strip('"'))
        i+=10
    #print(names)

    # now the surnames' turn 
    surnames.append(data_into_list[6].strip('"'))
    i=16
    while (i < len(data_into_list)):
        surnames.append(data_into_list[i].strip('"'))
        i+=10
    #print(surnames)

    return names,surnames


def getUsernames():
     # open the users.txt, fetch the data / content and converting it to a list " "
    f = open('users.txt', 'r')
    data = f.read()
    # replacing end of line('/n') with ' ' and
    # splitting the text it further when '.' is seen.
    data_into_list = data.replace('\n', ' ').split(" ")
    f.close()

    usernames = []

    usernames.append(data_into_list[1].strip('"')) # extracting passwords from data_into_list into the passwords list by getting rid of the (")s 
    i=11
    while (i < len(data_into_list)):
        usernames.append(data_into_list[i].strip('"'))
        i+=10
    print(usernames)
    return usernames


def identifyScenario(passwd):
    if (passwd == '1q2w3e4r5t6y7u8i9o0p*ğ-ü'):
        scenario = 0
    elif (passwd == '123456789'):
        scenario = 1
    elif (passwd == 'uiquiuiiuiuiuiuiuiuiuiuiui'):
        scenario = 2
    else:
        scenario = 3 # which indicates the good password
    return scenario

def identifyName_Surname(passwd, passwordIndex):
    names,surnames = getName_Surnames()
    # list to string
    name = ''.join(names[passwordIndex]) # name of the owner of the related password as string
    surname = ''.join(surnames[passwordIndex]) # surname of the owner
    #print(name+","+surname)

    # check if the first one or three letter of name or whole nameis in the password
    nStat,sStat = False, False
    if(
       name[0] in passwd or name[0].lower() in passwd or
       name[:3] in passwd or name[:3].lower() in passwd or
       name in passwd or name.lower() in passwd
       ):
        nStat = True
    
        
    # check the surname this time
    if(
       surname[0] in passwd or surname[0].lower() in passwd or
       surname[:3] in passwd or surname[:3].lower() in passwd or
       surname in passwd or surname.lower() in passwd
       ):
        sStat = True
    # now create the condition mechanism
    if(nStat == False and sStat == False):
        return 1 # which is the good situation
    else:
        return 0 # which indicates the bad situation
    
def identifyFeatureValues(passwords): # which identify the features of passwords taken as input
    passwordList = [*passwords]
    featureValues = []
    #print(passwordList)
    i=0
    while(i < len(passwordList)):
        featureValues.insert(i, [identifyName_Surname(passwordList[i],i), identifyScenario(passwordList[i])])
        i+=1
    #print("Feature List of passwords:\n"+str(featureValues))
    return featureValues


def AI_Logistic_Regression(passwordList):
    filepath = 'data.csv' # path for password file
    data = pandas.read_csv(filepath,',',error_bad_lines=False)
    data = pandas.DataFrame(data).values.astype('U')
    passwords = np.array(data) # here, converting the dataframe to a two-dimesional in other words, rank 2 array

    # Phase2: Data Preprocessing
    random.shuffle(passwords) #shuffling randomly for robustness

    # splitting the 'passwords' rank 2 array to two as 'y' which represents the columns of the passwords rank 2 array and as 'allpasswords' which represents the columns of the passwords rank 2 array 
    y = [d[1] for d in passwords] #labels (columns of the passwords rank 2 array)
    allpasswords= [d[0] for d in passwords] # actual passwords ((rows of the passwords rank 2 array)

    vectorizer = TfidfVectorizer(tokenizer=getTokens) # vectorization that is the process of transforming a token to a vector as a scope of linear algebria to prepare the data for logistic regression
    X = vectorizer.fit_transform(allpasswords) # transfroming the tokens to vectors
    
    # Phase3: Splitting Train dataset and test dataset through cross validation 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Phase4: Classification
    classifier = LogisticRegression(penalty='l2',multi_class='ovr')  # our logistic regression classifier
    classifier.fit(X_train, y_train) # training

    # More testing (Prediction Phase)
    X_predict = passwordList
    X_predict = vectorizer.transform(X_predict)
    y_Predict = classifier.predict(X_predict)
    #print(y_Predict)
    print('Logistic Regression Score: ' + str(classifier.score(X_test, y_test)))
    return y_Predict

def decisionTree():
    df = pandas.read_csv('passwd.csv') # read the train set
    df = pandas.DataFrame(df)
    #print(df)

    features = ['name_surname', 'attackScenario']
    X = df[features] # öznitelik sütunlarını, bir sonraki adımda karar verme işleminde kullanılmak üzere fit() metoduna bir argüman olarak vermek üzere bir değişkene atıyoruz.
    y = df['strength']

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X,y)

    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)

    graph.write_png('mydecisiontree.png') 
    img=pltimg.imread('mydecisiontree.png') 
    imgplot = plt.imshow(img) 
    plt.show()
    # the above image to be showed by screen indicates the road map our model will use it to predict and come to the decision.

    # declare our test features:
    featureList = identifyFeatureValues(getPasswords()) # which saves the features of all passwords
    print(featureList[0])
    tempList = []
    i = 0
    while(i < len(featureList)):
        tempList.append(dtree.predict([featureList[i]])) # it'll predict the strength which is our goal variable
        i+=1
    
    # convert the 2d array of results to a string
    resultStr = ''.join(str(item) for innerlist in tempList for item in innerlist)
    results = [*resultStr]
    #print(resultList)
    print('Decision Tree Score: ' + str(dtree.score(X,y)))
    return results # results in resultList indicate the strength value of each password taken as argument the decision tree!
    
    # print(dtree.predict([[1,0]]))
# test the algorithm


''' Password Checkers '''

def passwordCheckerv1(password):
    isWeak = True
    symCounter = 0
    letterCounter = 0
    numCounter = 0
    passwordList = [*password]

    i = 0
    while(i < len(passwordList)):
        if(32 <=ord(passwordList[i]) < 48 or 
           58 <=ord(passwordList[i]) < 65 or
           91 <=ord(passwordList[i]) < 97 or
           123 <=ord(passwordList[i]) < 127):
           symCounter += 1
        elif(65 <=ord(passwordList[i]) < 91 or 
           97 <=ord(passwordList[i]) < 123):
           letterCounter += 1
        elif(48 <= ord(passwordList[i]) < 58):
            numCounter += 1
        i += 1
        

    if(
       symCounter > 0 and
       letterCounter > 2 and
       numCounter > 2 and 
       len(passwordList) > 8
       ):
       isWeak = False # which indicates the good situation
    return isWeak
# Unit test of the v1:
def passwordCheckerv1_UnitTest(password, expected_answr):
    isWeak = passwordCheckerv1(password)
    if(str(isWeak) == expected_answr):
        print("Version1: Pass")
    else:
        print("Version1: Fail")


def passwordCheckerv2(password):
    isWeak = True
    symCounter = 0
    letterCounter = 0
    numCounter = 0
    passwordList = [*password]

    i = 0
    while(i < len(passwordList)):
        if(32 <=ord(passwordList[i]) < 48 or 
           58 <=ord(passwordList[i]) < 65 or
           91 <=ord(passwordList[i]) < 97 or
           123 <=ord(passwordList[i]) < 127):
           symCounter += 1
        elif(65 <=ord(passwordList[i]) < 91 or 
           97 <=ord(passwordList[i]) < 123):
           letterCounter += 1
        elif(48 <= ord(passwordList[i]) < 58):
            numCounter += 1
        i += 1
        
 
    j = 0
    c1 = False
    c2 = False
    proximity = False
    while(j < len(passwordList)):
        if(passwordList[j] == passwordList[(j+1)%len(passwordList)]):
           c1 = True
        elif(passwordList[j] == passwordList[(j+2)%len(passwordList)]):
           c2 = True
        if(c1 == True and c2 == True):
            proximity = True
        else:
            if(c1 == True and c2 == False):
                proximity = True
            elif(c1 == False and c2 == True):
                proximity = False
            else: 
                proximity = False
        j += 1

    if(
       symCounter > 0 and
       letterCounter > 2 and
       numCounter > 2 and 
       len(passwordList) > 8 and
       proximity == False
       ):
       isWeak = False
    return isWeak
# Unit test of the v2:
def passwordCheckerv2_UnitTest(password, expected_answr):
    isWeak = passwordCheckerv2(password)
    if(str(isWeak) == expected_answr):
        print("Version2: Pass")
    else:
        print("Version2: Fail")


def passwordCheckerv3(passwordList):
    strength = [] # 0: weak, 1: medium, 2: strong
    normal_results = []
    for password in passwordList:
        normal_results.append(passwordCheckerv2(password))
    #print(normal_results)
    log_reg_results = AI_Logistic_Regression(passwordList)
    d_tree_results = decisionTree()
    i=0
    while(i < len(passwordList)):
        if(
        (log_reg_results[i] == '2' and d_tree_results[i] == '1' and normal_results[i] == False) or
        (log_reg_results[i] == '2' and d_tree_results[i] == '1' and normal_results[i] == True)
        ):
            strength.append("2")
        elif(
        (log_reg_results[i] == '1' and d_tree_results[i] == '1' and normal_results[i] == False) or 
        (log_reg_results[i] == '0' and d_tree_results[i] == '1' and normal_results[i] == False) or
        (log_reg_results[i] == '1' and d_tree_results[i] == '0' and normal_results[i] == False) or
        (log_reg_results[i] == '2' and d_tree_results[i] == '0' and normal_results[i] == False) or
        (log_reg_results[i] == '2' and d_tree_results[i] == '1' and normal_results[i] == True) or
        (log_reg_results[i] == '1' and d_tree_results[i] == '1' and normal_results[i] == True)
        ):
            strength.append("1")
        elif(
        (log_reg_results[i] == '2' and d_tree_results[i] == '0' and normal_results[i] == True) or
        (log_reg_results[i] == '0' and d_tree_results[i] == '1' and normal_results[i] == True) or
        (log_reg_results[i] == '1' and d_tree_results[i] == '0' and normal_results[i] == True) or
        (log_reg_results[i] == '0' and d_tree_results[i] == '0' and normal_results[i] == False) or
        (log_reg_results[i] == '0' and d_tree_results[i] == '0' and normal_results[i] == True)
        ):
            strength.append("0")
        i+=1

    return strength    
# Unit test of the v3:
def passwordCheckerv3_UnitTest(password, expected_answr):
    strength = passwordCheckerv3(password)
    if (expected_answr == "strong"):
        expected_answr = 2
    elif (expected_answr == "medium"):
        expected_answr = 1
    elif (expected_answr == "weak"):
        expected_answr = 0

    if(strength == expected_answr):
        print("Version3: Pass")
    else:
        print("Version3: Fail, it's: " + str(strength))


''' Imports for 'detection logical location of weak passwords' function '''
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
#from selenium.webdriver import ActionChains
import pyautogui
import time

def identifyWeaks(strengthValuesOfPasswords, passwordList): 
    weaks = []
    i=0
    while(i< len(passwordList)):
        if (strengthValuesOfPasswords[i] == 0):
            weaks.append(passwordList[i])
        i+=1
    return weaks

def detectLogicalLocationOfWeaks(weaks):
    ''' Search for the usernames that corresponde these weak passwords in users.txt '''
    usernameList = getUsernames() # fetching the usernames corresponding our weak passwords from the users.txt
    passwordList = getPasswords() # fetching the passwords to compare in order to specify the corresponding usernames
    correspondingUserNames = []
    # create the correspondingUserNames list
    i=0
    while(i < len(passwordList)):
        if (passwordList[i] in weaks):
            correspondingUserNames.append(usernameList[i])
            #correspondingUserNames.append(usernameList[passwordList.index(passwordList[i])])
        i+=1

    ''' Pfsense Web Administrator Bot Creation Part '''
    # global declerations for Pfsense Web Administrator Login
    path = "C:\Program Files\chromedriver.exe"
    driver = webdriver.Chrome(path)
    driver.maximize_window()

    # the part of signing into the system 
    driver.get("http://"+hostname+"/")

    ''' In case of https '''
    # buttonAdvanced = driver.find_element(By.XPATH,'//*[@id="details-button"]')
    # buttonAdvanced.click()
    # buttonProceed = driver.find_element(By.XPATH,'//*[@id="proceed-link"]')
    # buttonProceed.click()

    # Sign in process;
    uNameOfPfsense = driver.find_element(By.XPATH,'//*[@id="usernamefld"]')
    uNameOfPfsense.send_keys(usernameOfPfsense) # entering / typing the username by selenium bot into related field

    passwdOfPfsense = driver.find_element(By.XPATH,'//*[@id="passwordfld"]')
    passwdOfPfsense.send_keys(passwordOfPfsense) # entering the password by selenium bot into related field
    passwdOfPfsense.send_keys(Keys.RETURN)
    time.sleep(3)

    # navigate the Status tab, and click it
    linkStatus = driver.find_element(By.XPATH,'//*[@id="pf-navbar"]/ul[1]/li[6]/a')
    linkStatus.click() # Status
    time.sleep(3)

    linkPortal = driver.find_element(By.XPATH,'//*[@id="pf-navbar"]/ul[1]/li[6]/ul/li[1]/a')
    linkPortal.click() # CaptivePortal
    time.sleep(3)

    selectDisplayZone = driver.find_element(By.XPATH,'//*[@id="zone"]')
    selectDisplayZone.click() # Select a Captive Portal Zone
    time.sleep(3) 

    optionStaff_Portal = driver.find_element(By.XPATH,'//*[@id="zone"]/option[2]')
    optionStaff_Portal.click() # Select a Captive Portal Zone Option
    time.sleep(3)
    
    # create an iteration to fetch and append each IP address of user into IPs list
    correspondingIPs = []
    i=0
    while (i < len(correspondingUserNames)):
        find = str(correspondingUserNames[i])

        # Use keys below
        pyautogui.hotkey('ctrl', 'f')

        # Write what you want
        pyautogui.write(find, interval=0.05)

        

        i+=1
    
    '''
    # create a variable for an object that performs action chains of our driver
    actChain = ActionChains(driver)
    # perform the Ctrl+f pressing action chain  
    actChain.key_down(Keys.CONTROL).send_keys('F').key_up(Keys.CONTROL).perform
    '''







''' Global Section of code '''
# fetchCredentials()
passwordList = getPasswords()

strengths = passwordCheckerv3(passwordList)
print('Result : ' + ','.join(strengths))




