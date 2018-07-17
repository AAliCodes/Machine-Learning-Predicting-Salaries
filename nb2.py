# -*- coding: utf-8 -*-
"""
This code aims to classify using naive bayes
"""
from __future__ import division
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import re
import string
import nltk
import csv
import pandas as pd

#number of distinct elements in each category before Kyle happened to the dataset
#age: 72 17
#working class: 8
#education : 16
#occupation : 15
#relationship : 6 [' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife']
#race : 5 [' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
#gender : 2 male, female
#work hours : 94
#native country : 41

#variables used in script
countOver = 0 
countUnder = 0
records = 0
#data structures holding prior counts, column 1:under = 50k, 
countAge = list()
Age = list()
countRace = list()
Race = list()
countEducation = list()
Education = list()
countGender = list()
Gender = list()
countWorkingHours = list()
WorkingHours = list()
countWorkingClass = list()
WorkingClass = list()
countOccupation = list()
Occupation = list()
#checks whether current row >/<
currentStatus = 0

#read csv into data structure
datafile = open('Data/train21k.csv','r') 
datareader = csv.reader(datafile, delimiter=",")
data = []
for row in datareader:
    data.append(row)
    records = records + 1
    #checks which class current row belongs to        
    if (">" in row[7]):
        countOver = countOver + 1
        currentStatus = 1
    else:
        countUnder = countUnder + 1 
        currentStatus = 0
    #age
    if row[0].strip() in Age:
        if currentStatus == 0:
            index = Age.index(row[0].strip())
            countAge[index][0] = countAge[index][0] + 1
        if currentStatus == 1:
            index = Age.index(row[0].strip())
            countAge[index][1] = countAge[index][1] + 1
    else:
        Age.append(row[0].strip())
        if currentStatus == 0:
            countAge.append([1,0])
        if currentStatus == 1:
            countAge.append([0,1])    
    #working class
    if row[1].strip() in WorkingClass:
        if currentStatus == 0:
            index = WorkingClass.index(row[1].strip())
            countWorkingClass[index][0] = countWorkingClass[index][0] + 1
        if currentStatus == 1:
            index = WorkingClass.index(row[1].strip())
            countWorkingClass[index][1] = countWorkingClass[index][1] + 1
    else:
        WorkingClass.append(row[1].strip())
        if currentStatus == 0:
            countWorkingClass.append([1,0])
        if currentStatus == 1:
            countWorkingClass.append([0,1])
    #Education
    if row[2].strip() in Education:
        if currentStatus == 0:
            index = Education.index(row[2].strip())
            countEducation[index][0] = countEducation[index][0] + 1
        if currentStatus == 1:
            index = Education.index(row[2].strip())
            countEducation[index][1] = countEducation[index][1] + 1
    else:
        Education.append(row[2].strip())
        if currentStatus == 0:
            countEducation.append([1,0])
        if currentStatus == 1:
            countEducation.append([0,1])
    #Occupation
    if row[3].strip() in Occupation:
        if currentStatus == 0:
            index = Occupation.index(row[3].strip())
            countOccupation[index][0] = countOccupation[index][0] + 1
        if currentStatus == 1:
            index = Occupation.index(row[3].strip())
            countOccupation[index][1] = countOccupation[index][1] + 1
    else:
        Occupation.append(row[3].strip())
        if currentStatus == 0:
            countOccupation.append([1,0])
        if currentStatus == 1:
            countOccupation.append([0,1])
    #Race
    if row[4].strip() in Race:
        if currentStatus == 0:
            index = Race.index(row[4].strip())
            countRace[index][0] = countRace[index][0] + 1
        if currentStatus == 1:
            index = Race.index(row[4].strip())
            countRace[index][1] = countRace[index][1] + 1
    else:
        Race.append(row[4].strip())
        if currentStatus == 0:
            countRace.append([1,0])
        if currentStatus == 1:
            countRace.append([0,1])
    #Gender
    if row[5].strip() in Gender:
        if currentStatus == 0:
            index = Gender.index(row[5].strip())
            countGender[index][0] = countGender[index][0] + 1
        if currentStatus == 1:
            index = Gender.index(row[5].strip())
            countGender[index][1] = countGender[index][1] + 1
    else:
        Gender.append(row[5].strip())
        if currentStatus == 0:
            countGender.append([1,0])
        if currentStatus == 1:
            countGender.append([0,1])
    #Weekly work hours
    if row[6].strip() in WorkingHours:
        if currentStatus == 0:
            index = WorkingHours.index(row[6].strip())
            countWorkingHours[index][0] = countWorkingHours[index][0] + 1
        if currentStatus == 1:
            index = WorkingHours.index(row[6].strip())
            countWorkingHours[index][1] = countWorkingHours[index][1] + 1
    else:
        WorkingHours.append(row[6].strip())
        if currentStatus == 0:
            countWorkingHours.append([1,0])
        if currentStatus == 1:
            countWorkingHours.append([0,1])
#output
#print(records)
'''           
print(Age) #UBUDALA
print(countAge)
print(Education)#imfundo
print(countEducation)
print(Occupation)
print(countOccupation)
print(Race)
print(countRace)
print(Gender)
print(countGender)
print(WorkingClass)
print(countWorkingClass)
print(WorkingHours)
print(countWorkingHours)
'''
#insert test data
#Test = ['17-34','Private','Some-college','Machine-op-inspct','Amer-Indian-Eskimo','Male','40-59','<=50K']
#Test = ['50-65','Private','School','Craft-repair','White','Male','0-19','<=50K']










#do naive bayes
pAgeOver = 0
pEducationOver = 0
pOccupationOver = 0
pRaceOver = 0
pGenderOver = 0
pWorkingClassOver = 0
pWorkingHoursOver = 0

#testing data
testOver = 0
testUnder = 0

predOver = 0
predUnder = 0

predLogOver = 0
predLogUnder = 0

truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

LogtruePos = 0
LogtrueNeg = 0
LogfalsePos = 0
LogfalseNeg = 0

Accuracy = 0;
logAccuracy = 0;


file = open('Data/test9k.csv','r')
reader = csv.reader(file, delimiter=",")
test = []
for Test in reader:
    test.append(Test)
    pPos = 0
    pNeg = 0
    pLogPos = 0
    pLogNeg = 0
    
    #checks which class current row belongs to        
    if (">" in Test[7]):
        testOver += 1
        currentStatus = 0
    else:
        testUnder +=  1
        currentStatus = 1
   
        
    #age
    if Test[0].strip() in Age:
        if currentStatus == 0:
            index = Age.index(Test[0].strip())
            countAge[index][0] = countAge[index][0] + 1
        if currentStatus == 1:
            index = Age.index(Test[0].strip())
            countAge[index][1] = countAge[index][1] + 1
    else:
        Age.append(Test[0].strip())
        if currentStatus == 0:
            countAge.append([1,0])
        if currentStatus == 1:
            countAge.append([0,1])    
    #pAgeUnder = round((countAge[Age.index(Test[0].strip())][0]/(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0])),2)
    #pAgeOver = round((countAge[Age.index(Test[0].strip())][1]/(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0])),2)
    pLogAgeOver = np.log(countAge[Age.index(Test[0].strip())][1])-np.log(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0])
    pLogAgeUnder = np.log(countAge[Age.index(Test[0].strip())][0])-np.log(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0]) 
    
    #working class
    if Test[1].strip() in WorkingClass:
        if currentStatus == 0:
            index = WorkingClass.index(Test[1].strip())
            countWorkingClass[index][0] = countWorkingClass[index][0] + 1
        if currentStatus == 1:
            index = WorkingClass.index(Test[1].strip())
            countWorkingClass[index][1] = countWorkingClass[index][1] + 1
    else:
        WorkingClass.append(Test[1].strip())
        if currentStatus == 0:
            countWorkingClass.append([1,0])
        if currentStatus == 1:
            countWorkingClass.append([0,1])
    #pWorkingClassUnder = round((countWorkingClass[WorkingClass.index(Test[1].strip())][0]/(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])),2)
    #pWorkingClassOver = round((countWorkingClass[WorkingClass.index(Test[1].strip())][1]/(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])),2)
    pLogWorkingClassOver = np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][1]) - np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])
    pLogWorkingClassUnder = np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][0]) - np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])

    #Education
    if Test[2].strip() in Education:
        if currentStatus == 0:
            index = Education.index(Test[2].strip())
            countEducation[index][0] = countEducation[index][0] + 1
        if currentStatus == 1:
            index = Education.index(Test[2].strip())
            countEducation[index][1] = countEducation[index][1] + 1
    else:
        Education.append(Test[2].strip())
        if currentStatus == 0:
            countEducation.append([1,0])
        if currentStatus == 1:
            countEducation.append([0,1])
    #pEducationUnder = round((countEducation[Education.index(Test[2].strip())][0]/(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])),2)
    #pEducationOver = round((countEducation[Education.index(Test[2].strip())][1]/(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])),2)
    pLogEducationOver = np.log(countEducation[Education.index(Test[2].strip())][1]) - np.log(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])
    pLogEducationUnder = np.log(countEducation[Education.index(Test[2].strip())][0]) - np.log(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])

    #Occupation
    if Test[3].strip() in Occupation:
        if currentStatus == 0:
            index = Occupation.index(Test[3].strip())
            countOccupation[index][0] = countOccupation[index][0] + 1
        if currentStatus == 1:
            index = Occupation.index(row[3].strip())
            countOccupation[index][1] = countOccupation[index][1] + 1
    else:
        Occupation.append(row[3].strip())
        if currentStatus == 0:
            countOccupation.append([1,0])
        if currentStatus == 1:
            countOccupation.append([0,1])
    #pOccupationUnder = round((countOccupation[Occupation.index(Test[3].strip())][0]/(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])),2)
    #pOccupationOver = round((countOccupation[Occupation.index(Test[3].strip())][1]/(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])),2)
    pLogOccupationOver = np.log(countOccupation[Occupation.index(Test[3].strip())][1]) - np.log(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])
    pLogOccupationUnder = np.log(countOccupation[Occupation.index(Test[3].strip())][0]) - np.log(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])
     
    #Race
    if row[4].strip() in Race:
        if currentStatus == 0:
            index = Race.index(row[4].strip())
            countRace[index][0] = countRace[index][0] + 1
        if currentStatus == 1:
            index = Race.index(row[4].strip())
            countRace[index][1] = countRace[index][1] + 1
    else:
        Race.append(row[4].strip())
        if currentStatus == 0:
            countRace.append([1,0])
        if currentStatus == 1:
            countRace.append([0,1])
    #pRaceUnder = round((countRace[Race.index(Test[4].strip())][0]/(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])),2)
    #pRaceOver = round((countRace[Race.index(Test[4].strip())][1]/(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])),2)
    pLogRaceOver = np.log(countRace[Race.index(Test[4].strip())][1]) - np.log(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])
    pLogRaceUnder = np.log(countRace[Race.index(Test[4].strip())][0]) - np.log(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])

    #Gender
    if row[5].strip() in Gender:
        if currentStatus == 0:
            index = Gender.index(row[5].strip())
            countGender[index][0] = countGender[index][0] + 1
        if currentStatus == 1:
            index = Gender.index(row[5].strip())
            countGender[index][1] = countGender[index][1] + 1
    else:
        Gender.append(row[5].strip())
        if currentStatus == 0:
            countGender.append([1,0])
        if currentStatus == 1:
            countGender.append([0,1])
    #pGenderUnder = round((countGender[Gender.index(Test[5].strip())][0]/(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])),2)
    #pGenderOver = round((countGender[Gender.index(Test[5].strip())][1]/(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])),2)
    pLogGenderOver = np.log(countGender[Gender.index(Test[5].strip())][1]) - np.log(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])
    pLogGenderUnder = np.log(countGender[Gender.index(Test[5].strip())][0]) - np.log(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])

    #Weekly work hours
    if row[6].strip() in WorkingHours:
        if currentStatus == 0:
            index = WorkingHours.index(row[6].strip())
            countWorkingHours[index][0] = countWorkingHours[index][0] + 1
        if currentStatus == 1:
            index = WorkingHours.index(row[6].strip())
            countWorkingHours[index][1] = countWorkingHours[index][1] + 1
    else:
        WorkingHours.append(row[6].strip())
        if currentStatus == 0:
            countWorkingHours.append([1,0])
        if currentStatus == 1:
            countWorkingHours.append([0,1])
    #pWorkingHoursUnder = round((countWorkingHours[WorkingHours.index(Test[6].strip())][0]/(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])),2)
    #pWorkingHoursOver = round((countWorkingHours[WorkingHours.index(Test[6].strip())][1]/(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])),2)
    pLogWorkingHoursOver = np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][1]) - np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])
    pLogWorkingHoursUnder = np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][0]) - np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])

    logProbabilityOver = (pLogAgeOver + pLogRaceOver + pLogEducationOver + pLogGenderOver + pLogWorkingHoursOver + pLogWorkingClassOver + pLogOccupationOver + pLogOver) - np.log(((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder)))
    logProbabilityUnder = (pLogAgeUnder + pLogRaceUnder + pLogEducationUnder + pLogGenderUnder + pLogWorkingHoursUnder + pLogWorkingClassUnder + pLogOccupationUnder + pLogUnder) - np.log(((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder)))

    #probabilityOver = (pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)/((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder))
    #probabilityUnder = (pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder)/((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder))
    '''
    if probabilityOver > probabilityUnder:
        predOver += 1
        if (">" in Test[7]):
            truePos += 1
        else:
            falsePos += 1
    else: #calculated under 50k
        predUnder += 1
        if (">" in Test[7]):
            falseNeg += 1
        else:
            trueNeg += 1
'''   
            
    #logs
    if logProbabilityOver > logProbabilityUnder:
        predLogOver += 1
        if (">" in Test[7]):
            LogtruePos += 1
        else:
            LogfalsePos += 1
    else: #calculated under 50k
        predLogUnder += 1
        if ("<" in Test[7]):
            LogfalseNeg += 1
        else:
            LogtrueNeg += 1
                
        
            
            
Accuracy = round(((truePos + trueNeg)/(testOver+testUnder))*100,2)
logAccuracy = round(((LogtruePos + LogtrueNeg)/(testOver+testUnder))*100,2)

print("Log accuracy -", logAccuracy)
print(" \n")            
print("****confusion matrix:****")
print("      ||   TRUE   || FALSE||" )
print("TRUE  ||  ", LogtrueNeg, "  ||",LogfalsePos, " || " )
print("FALSE ||  ", LogfalseNeg, "   ||",LogtruePos, "  ||" )

print(records)

'''            

print(logProbabilityOver)
print(logProbabilityUnder)
    
print(probabilityOver)
print(probabilityUnder)

print("true test")        
print(testOver , "were really over")
print(testUnder, 'were really under')
print(" ")

print("pred over and under")
print(predOver, 'predicted over')
print(predUnder, 'predicted under')
print(" ")

print("log result")
print(LogtruePos, 'true over')
print(LogtrueNeg, 'true under')
print(LogfalsePos, ' false over')
print(LogfalseNeg, 'false under')
print(" ")



print("results")
print(truePos, 'true over')
print(trueNeg,'true under')
print(falsePos, 'false over')
print(falseNeg, 'false under')
'''
#Kyle
#print(Race.index(Test[4].strip()))

#under

#pAgeUnder = round((countAge[Age.index(Test[0].strip())][0]/(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0])),2)
#pWorkingClassUnder = round((countWorkingClass[WorkingClass.index(Test[1].strip())][0]/(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])),2)
#pEducationUnder = round((countEducation[Education.index(Test[2].strip())][0]/(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])),2)
#OccupationUnder = round((countOccupation[Occupation.index(Test[3].strip())][0]/(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])),2)
#pRaceUnder = round((countRace[Race.index(Test[4].strip())][0]/(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])),2)
#pGenderUnder = round((countGender[Gender.index(Test[5].strip())][0]/(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])),2)
#pWorkingHoursUnder = round((countWorkingHours[WorkingHours.index(Test[6].strip())][0]/(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])),2)

#over

#pAgeOver = round((countAge[Age.index(Test[0].strip())][1]/(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0])),2)
#pWorkingClassOver = round((countWorkingClass[WorkingClass.index(Test[1].strip())][1]/(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])),2)
#pEducationOver = round((countEducation[Education.index(Test[2].strip())][1]/(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])),2)
#pOccupationOver = round((countOccupation[Occupation.index(Test[3].strip())][1]/(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])),2)
#pRaceOver = round((countRace[Race.index(Test[4].strip())][1]/(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])),2)
#pGenderOver = round((countGender[Gender.index(Test[5].strip())][1]/(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])),2)
#pWorkingHoursOver = round((countWorkingHours[WorkingHours.index(Test[6].strip())][1]/(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])),2)

#probabilities over
#pLogAgeOver = np.log(countAge[Age.index(Test[0].strip())][1])-np.log(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0])
#pLogWorkingClassOver = np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][1]) - np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])
#pLogEducationOver = np.log(countEducation[Education.index(Test[2].strip())][1]) - np.log(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])
#pLogOccupationOver = np.log(countOccupation[Occupation.index(Test[3].strip())][1]) - np.log(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])
#pLogRaceOver = np.log(countRace[Race.index(Test[4].strip())][1]) - np.log(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])
#pLogGenderOver = np.log(countGender[Gender.index(Test[5].strip())][1]) - np.log(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])
#pLogWorkingHoursOver = np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][1]) - np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])

#probabilities under
#pLogAgeUnder = np.log(countAge[Age.index(Test[0].strip())][0])-np.log(countAge[Age.index(Test[0].strip())][1] + countAge[Age.index(Test[0].strip())][0])
#pLogWorkingClassUnder = np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][0]) - np.log(countWorkingClass[WorkingClass.index(Test[1].strip())][1] + countWorkingClass[WorkingClass.index(Test[1].strip())][0])
#pLogEducationUnder = np.log(countEducation[Education.index(Test[2].strip())][0]) - np.log(countEducation[Education.index(Test[2].strip())][1] + countEducation[Education.index(Test[2].strip())][0])
#pLogOccupationUnder = np.log(countOccupation[Occupation.index(Test[3].strip())][0]) - np.log(countOccupation[Occupation.index(Test[3].strip())][1] + countOccupation[Occupation.index(Test[3].strip())][0])
#pLogRaceUnder = np.log(countRace[Race.index(Test[4].strip())][0]) - np.log(countRace[Race.index(Test[4].strip())][1] + countRace[Race.index(Test[4].strip())][0])
#pLogGenderUnder = np.log(countGender[Gender.index(Test[5].strip())][0]) - np.log(countGender[Gender.index(Test[5].strip())][1] + countGender[Gender.index(Test[5].strip())][0])
#pLogWorkingHoursUnder = np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][0]) - np.log(countWorkingHours[WorkingHours.index(Test[6].strip())][1] + countWorkingHours[WorkingHours.index(Test[6].strip())][0])

#probabilities <=50k and >50k

pLogUnder = np.log(countUnder) - np.log(countUnder + countOver)
pLogOver = np.log(countOver) - np.log(countUnder + countOver)

pUnder = (countUnder/(countUnder+countOver))
pOver = (countOver/(countUnder+countOver))

'''

logProbabilityOver = (pLogAgeOver + pLogRaceOver + pLogEducationOver + pLogGenderOver + pLogWorkingHoursOver + pLogWorkingClassOver + pLogOccupationOver + pLogOver) - np.log(((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder)))
logProbabilityUnder = (pLogAgeUnder + pLogRaceUnder + pLogEducationUnder + pLogGenderUnder + pLogWorkingHoursUnder + pLogWorkingClassUnder + pLogOccupationUnder + pLogUnder) - np.log(((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder)))

probabilityOver = (pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)/((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder))
probabilityUnder = (pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder)/((pAgeOver*pRaceOver*pEducationOver*pGenderOver*pWorkingHoursOver*pWorkingClassOver*pOccupationOver*pOver)+(pAgeUnder*pRaceUnder*pEducationUnder*pGenderUnder*pWorkingHoursUnder*pWorkingClassUnder*pOccupationUnder*pUnder))
'''

'''

print("Above probs")
print(pAgeOver)
print(pEducationOver)
print(pOccupationOver)
print(pRaceOver)
print(pGenderOver)
print(pWorkingClassOver)
print(pWorkingHoursOver)
print("Below probs")
print(pAgeUnder)
print(pEducationUnder)
print(pOccupationUnder)
print(pRaceUnder)
print(pGenderUnder)
print(pWorkingClassUnder)
print(pWorkingHoursUnder)
print("probabilities")
print(probabilityOver)
print(probabilityUnder)
print("log probabilities")
print(logProbabilityOver)
print(logProbabilityUnder)
'''
"""
print(pEqualOrUnder)
print(pOver)

print(countUnder/(countOver+countUnder))
print(countOver/(countOver+countUnder))


print(probabilityOver)
"""






