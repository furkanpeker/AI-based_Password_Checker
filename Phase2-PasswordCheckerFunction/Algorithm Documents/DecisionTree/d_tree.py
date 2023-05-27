import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from array import *

''' Check if we're in the correct directory:
import os

# get current working directory
cwd = os.getcwd()

#get files in directory
files = os.listdir(cwd) 

print(files)
'''

# Auxilary Functions for the Decision Tree Algrorithm:
def getPasswords():
    # open the users.txt and converting its content into a list " "
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

    return passwords

def getName_Surnames():
    # open the users.txt and converting its content into a list " "
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
    print("Feature List of passwords:\n"+str(featureValues))
    return featureValues

# Decision Tree Function:
def decisionTree():
    df = pandas.read_csv('passwd.csv') # read the train set
    df = pandas.DataFrame(df)
    print(df)

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
    #print(featureList[0])
    tempList = []
    i = 0
    while(i < len(featureList)):
        tempList.append(dtree.predict([featureList[i]])) # it'll predict the strength which is our goal variable
        i+=1
    
    # convert the 2d array of results to a string
    results = ','.join(str(item) for innerlist in tempList for item in innerlist)


    print(results)
    return results # results in resultList indicate the strength value of each password taken as argument the decision tree!
    
    # print(dtree.predict([[1,0]]))
    
# test the algorithm
decisionTree()
