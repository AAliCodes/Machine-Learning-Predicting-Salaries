# -*- coding: utf-8 -*-
"""
This code aims to classify using decision trees
"""
from __future__ import division
import numpy as np
import csv
import pandas as pd
import math
import copy

#variables used in script
countOver = 0 
countUnder = 0
records = 0
nodeNumber = 1
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
currentRecords = 0
currentCountOver = 0
currentCountUnder = 0
currentOverProb = 0
currentUnderProb = 0

#get attributes and counts of a dataset
def getStats(current):
    global currentStatus
    global currentRecords
    global currentCountOver
    global currentCountUnder
    global countAge
    global Age
    global countRace
    global Race
    global countEducation
    global Education 
    global countGender
    global Gender
    global countWorkingHours
    global WorkingHours
    global countWorkingClass
    global WorkingClass
    global countOccupation
    global Occupation
    global currentOverProb
    global currentUnderProb
    
    currentOverProb = 0 
    currentUnderProb = 0     
    currentRecords = 0
    currentCountOver = 0
    currentCountUnder = 0
    countAge.clear()
    Age.clear()
    countRace.clear()
    Race.clear()
    countEducation.clear()
    Education.clear()
    countGender.clear()
    Gender.clear()
    countWorkingHours.clear()
    WorkingHours.clear()
    countWorkingClass.clear()
    WorkingClass.clear()
    countOccupation.clear()
    Occupation.clear()
    
    for i in range(len(current)):
        currentRecords = currentRecords + 1
        #checks which class current row belongs to        
        if (">" in current[i][7]):
            currentCountOver = currentCountOver + 1
            currentStatus = 1
        else:
            currentCountUnder = currentCountUnder + 1 
            currentStatus = 0
        #age
        if current[i][0].strip() in Age:
            if currentStatus == 0:
                index = Age.index(current[i][0].strip())
                countAge[index][0] = countAge[index][0] + 1
            if currentStatus == 1:
                index = Age.index(current[i][0].strip())
                countAge[index][1] = countAge[index][1] + 1
        else:
            Age.append(current[i][0].strip())
            if currentStatus == 0:
                countAge.append([1,0])
            if currentStatus == 1:
                countAge.append([0,1])    
        #working class
        if current[i][1].strip() in WorkingClass:
            if currentStatus == 0:
                index = WorkingClass.index(current[i][1].strip())
                countWorkingClass[index][0] = countWorkingClass[index][0] + 1
            if currentStatus == 1:
                index = WorkingClass.index(current[i][1].strip())
                countWorkingClass[index][1] = countWorkingClass[index][1] + 1
        else:
            WorkingClass.append(current[i][1].strip())
            if currentStatus == 0:
                countWorkingClass.append([1,0])
            if currentStatus == 1:
                countWorkingClass.append([0,1])
        #Education
        if current[i][2].strip() in Education:
            if currentStatus == 0:
                index = Education.index(current[i][2].strip())
                countEducation[index][0] = countEducation[index][0] + 1
            if currentStatus == 1:
                index = Education.index(current[i][2].strip())
                countEducation[index][1] = countEducation[index][1] + 1
        else:
            Education.append(current[i][2].strip())
            if currentStatus == 0:
                countEducation.append([1,0])
            if currentStatus == 1:
                countEducation.append([0,1])
        #Occupation
        if current[i][3].strip() in Occupation:
            if currentStatus == 0:
                index = Occupation.index(current[i][3].strip())
                countOccupation[index][0] = countOccupation[index][0] + 1
            if currentStatus == 1:
                index = Occupation.index(current[i][3].strip())
                countOccupation[index][1] = countOccupation[index][1] + 1
        else:
            Occupation.append(current[i][3].strip())
            if currentStatus == 0:
                countOccupation.append([1,0])
            if currentStatus == 1:
                countOccupation.append([0,1])
        #Race
        if current[i][4].strip() in Race:
            if currentStatus == 0:
                index = Race.index(current[i][4].strip())
                countRace[index][0] = countRace[index][0] + 1
            if currentStatus == 1:
                index = Race.index(current[i][4].strip())
                countRace[index][1] = countRace[index][1] + 1
        else:
            Race.append(current[i][4].strip())
            if currentStatus == 0:
                countRace.append([1,0])
            if currentStatus == 1:
                countRace.append([0,1])
        #Gender
        if current[i][5].strip() in Gender:
            if currentStatus == 0:
                index = Gender.index(current[i][5].strip())
                countGender[index][0] = countGender[index][0] + 1
            if currentStatus == 1:
                index = Gender.index(current[i][5].strip())
                countGender[index][1] = countGender[index][1] + 1
        else:
            Gender.append(current[i][5].strip())
            if currentStatus == 0:
                countGender.append([1,0])
            if currentStatus == 1:
                countGender.append([0,1])
        #Weekly work hours
        if current[i][6].strip() in WorkingHours:
            if currentStatus == 0:
                index = WorkingHours.index(current[i][6].strip())
                countWorkingHours[index][0] = countWorkingHours[index][0] + 1
            if currentStatus == 1:
                index = WorkingHours.index(current[i][6].strip())
                countWorkingHours[index][1] = countWorkingHours[index][1] + 1
        else:
            WorkingHours.append(current[i][6].strip())
            if currentStatus == 0:
                countWorkingHours.append([1,0])
            if currentStatus == 1:
                countWorkingHours.append([0,1])
                   
        currentOverProb = currentCountOver/(currentRecords)
        currentUnderProb = currentCountUnder/(currentRecords)
        
#calculate entropy
def Entropy(feature, countFeature):
    entropy = 0 
    countAbove = 0
    countBelow = 0
    totalCount = 0
    size = len(feature)
    for i in range(size):
        countBelow = countBelow + countFeature[i][0]
        countAbove = countAbove + countFeature[i][1]
    totalCount = countAbove + countBelow
    
    pAbove = (countAbove/totalCount)
    pBelow = (countBelow/totalCount)    
    entropy = ( -(pAbove * (np.log2(pAbove))) -(pBelow * (np.log2(pBelow)))).round(3)
    return entropy;

#gain of an attribute
def Gain(feature, countFeature):
    if(len(feature) > 0):
        HD = 0
        HD = Entropy(feature, countFeature)
        weight = 0
        temp = 0
        for i in range(len(feature)):
            currentWeight = countFeature[i][0] + countFeature[i][1]
            weight = weight + currentWeight
            pAbove = countFeature[i][0]/currentWeight
            pBelow = countFeature[i][1]/currentWeight
            if pAbove == 0:
                temp = temp + (currentWeight*(-(pBelow*np.log2(pBelow))))
            elif pBelow == 0:
                temp = temp + (currentWeight*( - (pAbove*np.log2(pAbove))))
            else:
                temp = temp + (currentWeight*(-(pBelow*np.log2(pBelow)) - (pAbove*np.log2(pAbove))))
        if temp>0:
            Gain = HD - ((1/weight)*temp)
        else:
            Gain = 0
    else:
        Gain = 0
    return Gain;

#best feature
def BestFeature(featureLists, countLists):
    index = 0
    currentGain = 0
    for i in range(len(featureLists)):
        gain = Gain(featureLists[i],countLists[i])
        if gain > currentGain:
            index = i
            currentGain = gain
    return index  
#node stuff
class Node:
    def __init__(self,data):
        self.data = data
        self.parent = None
        self.children = None
        self.branch = None
        self.decision = None
        self.nodeNumber = 0
    def setNodeNumber(self,nodeNumber):
        self.nodeNumber = nodeNumber
    def getNodeNumber(self):
        return self.nodeNumber
    def setData(self,new):
        self.data = new
    def getData(self):
        return self.data
    def setParent(self,parent):
        self.parent = parent
    def getParent(self):
        return self.parent
    def setChildren(self,children):
        self.children = children
    def getChildren(self):
        return self.children
    def setBranch(self,branch):
        self.branch = branch
    def getBranch(self):
        return self.branch
    def setDecision(self,decision):
        self.decision = decision
    def getDecision(self):
        return self.decision
    def printTree(self, level=0):
        if self.branch != None and self.decision != None:
            print( '\t'*level + ' question: ' + self.branch + ' decision: ',self.decision)
        elif self.decision != None:
            print( '\t'*level + ' classification: ',self.decision)
        elif self.branch != None:
            print('\t'*level + ' question: ' + self.branch)            
        if self.children != None:
            for child in self.children:
                child.printTree(level+1)

#Gets child nodes of node passed
def getChildren(currentNode):
    global bestFeature
    global bestFeatureCount
    global tempData
    currentData = copy.deepcopy(currentNode.getData())
    currentChildrenList = list()    
    for i in range(len(bestFeature)):
        del tempData
        tempData = []
        for j in range(len(currentData)):
            if currentData[j][bestFeatureIndex].strip() == bestFeature[i]:
                tempData.append(currentData[j])
        tempNode = Node(tempData)
        tempNode.setBranch(bestFeature[i])
        currentChildrenList.append(tempNode)
    return currentChildrenList;  

#Builds Tree recursively
def buildTree(currentNode):      
    decision = None
    global bestFeature
    global bestFeatureIndex
    global depth
    global currentRecords
    global currentStatus
    global currentRecords
    global currentCountOver
    global currentCountUnder
    global countAge
    global Age
    global countRace
    global Race
    global countEducation
    global Education 
    global countGender
    global Gender
    global countWorkingHours
    global WorkingHours
    global countWorkingClass
    global WorkingClass
    global countOccupation
    global Occupation
    global currentOverProb
    global currentUnderProb
    global rootNode
      
    getStats(currentNode.getData())
    if currentOverProb>=0.75:
        decision = 1
        currentNode.setDecision(decision)
    elif currentUnderProb>=0.75:
        decision = 0
        currentNode.setDecision(decision)
    else:
        getStats(currentNode.getData())
        ListOfLists = [Age,WorkingClass,Education,Occupation,Race,Gender,WorkingHours]
        ListOfCountLists = [countAge,countWorkingClass,countEducation,countOccupation,countRace,countGender,countWorkingHours]
        bestFeatureIndex = BestFeature(ListOfLists,ListOfCountLists)
        bestFeature  = ListOfLists[bestFeatureIndex]     
        childrenList = copy.deepcopy(getChildren(currentNode))
        currentNode.setChildren(childrenList)
        for i in range(len(childrenList)):
            if currentNode.getBranch() == childrenList[i].getBranch():
#                return
                childrenList[i].setDecision(None)
            else:
                buildTree(childrenList[i])            
    return
#read csv into data structure for training
datafile = open('train.csv','r')
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

#ID3
ListOfLists = [Age,WorkingClass,Education,Occupation,Race,Gender,WorkingHours]
ListOfCountLists = [countAge,countWorkingClass,countEducation,countOccupation,countRace,countGender,countWorkingHours]
bestFeatureIndex = BestFeature(ListOfLists,ListOfCountLists)
bestFeature  = ListOfLists[bestFeatureIndex]   
rootNode = Node(data)
buildTree(rootNode)
rootNode.printTree()

#Testing
datafile = open('Data/test.csv','r')
datareader = csv.reader(datafile, delimiter=",")
data = []
classifications = list()
predictions = list()

trueNegative = 0
truePositive = 0
falseNegative = 0
falsePositive = 0


if(countOver/records)>(countUnder/records):
    defaultProb = 1
else:
    defaultProb = 0

for row in datareader:
    data.append(row)
    #checks which class current row belongs to        
    if (">" in row[7]):
        classifications.append(1)
        currentStatus = 1
    else:
        classifications.append(0)
        currentStatus = 0

    testNode = copy.deepcopy(rootNode)
    prediction = None
    found = False
    while (testNode.getChildren() != None) and (prediction == None):
        found = False
        currentChildren = testNode.getChildren()
        for i in range(len(currentChildren)):
            for j in range(len(row)):
                if currentChildren[i].getBranch() == row[j].strip():
                    found = True
                    testNode = testNode.getChildren()[i]
                    prediction = testNode.getDecision()
        if found == False:
            prediction = defaultProb
    if prediction == None or prediction == 'None':
        prediction = defaultProb 
    predictions.append(prediction)
    if(prediction==0) and (currentStatus==0):
        trueNegative+=1        
    elif(prediction==1) and (currentStatus==1):
        truePositive+=1
    elif(prediction==0) and (currentStatus==1):
        falseNegative+=1
    elif(prediction==1) and (currentStatus==0):
        falsePositive+=1
        
    print(len(predictions))

print("****confusion matrix:****")
print("      ||   TRUE   || FALSE||" )
print("TRUE  ||  ", trueNegative, "   ||",falsePositive, " || " )
print("FALSE ||  ", falseNegative, "   ||",truePositive, "  ||" )
same = 0
for i in range(len(classifications)):
    if classifications[i] == predictions[i]:
        same+=1
    
print('the accuracy is: ',(same/(len(predictions))*100),'%')

                



