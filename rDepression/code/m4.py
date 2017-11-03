#m4.py
"""Contains our implementation of Mixed Membership Markov Models (M4) with their
training and prediction algorithms. To learn more about M4, feel free to look at
http://cmci.colorado.edu/~mpaul/files/emnlp2012-m4.pdf ."""

#imports

import numpy as np
import pandas as pd
from discourseGraph import DiscourseGraph, DiscourseNode
import pickle as pkl

#classes

class m4:
    """Class that holds our implementation of Mixed Membership Markov Models
    (M4)."""

    def __init__(self,numClasses,transitionWeightVar = 10,
                 dirichletScalePar = 1):
        self.numClasses = numClasses
        self.transitionWeightVar = transitionWeightVar
        self.dirichletScalePar = dirichletScalePar
        #initialize class transition matrix
        self.classTransitionMat = np.random.normal(loc = 0,
                                      scale = np.sqrt(self.transitionWeightVar),
                                      size = (numClasses,numClasses + 2))
    
    def initDiscourse(self,discourseNode,isParent):
        """Helper for initializing the feature vector of a given discourse node
        and then forming the vocabulary present from the discourse succeeding
        said discourse node"""
        #create feature vector
        discourseNode.featureVec = np.zeros(self.numClasses + 2)
        discourseNode.featureVec[self.numClasses + 1] = 1 #bias feature
        if (isParent):
            discourseNode.featureVec[self.numClasses] = 1 #parent indicator
        #then initialize random topic assignment
        discourseNode.topicAssignment = np.random.randint(self.numClasses,
                                            size = len(discourseNode.tokens))
        #then finalize feature vector
        for class_ in range(self.numClasses):
            discourseNode.featureVec[class_] = np.sum(
                             discourseNode.topicAssignment == class_)
        #then initialize discourse vocab
        discourseVocab = {}
        for class_ in range(self.numClasses):
            discourseVocab[class_] = {}
        for word in discourseNode.tokens:
            for class_ in range(self.numClasses):
                discourseVocab[class_][word] = 0
        #then populate discourse vocab
        for i in range(discourseNode.topicAssignment.shape[0]):
            givenWord = discourseNode.tokens[i]
            givenClass = discourseNode.topicAssignment[i]
            discourseVocab[givenClass][givenWord] += 1
        #then get each child's vocab
        for child in discourseNode.children:
            childVocab = self.initDiscourse(child,False)
            for class_ in childVocab:
                givenClassChildVocab = childVocab[class_]
                for word in givenClassChildVocab:
                    if (word in discourseVocab[class_]): #update
                        discourseVocab[class_][word] += givenClassChildVocab[
                                                                        word]
                    else: #initialize
                        discourseVocab[class_][word] = givenClassChildVocab[
                                                                        word]
        return discourseVocab 

    def formVocabularyAndFeatures(self,discourseGraphList):
        """Helper for getting the vocabulary and feature vectors for each
        of our discourses"""
        #initialize vocabulary
        vocabDict = {}
        for class_ in range(self.numClasses):
            vocabDict[class_] = {} #will add to these
        for discourseGraph in discourseGraphList:
            givenDiscourseVocab = self.initDiscourse(discourseGraph.root,True)
            #update our main vocabulary
            for class_ in givenDiscourseVocab:
                givenClassVocab = givenDiscourseVocab[class_]
                for word in givenClassVocab:
                    if (word in vocabDict[class_]): #update
                        vocabDict[class_][word] += givenClassVocab[word]
                    else: #initialize
                        vocabDict[class_][word] = givenClassVocab[word]
        return vocabDict

    def initializePars(self,discourseGraphList):
        """helper for preparing our dataset for M4 fitting"""
        #get vocabulary frame after forming feature vectors for each text in
        #each discourse
        self.vocabDict = self.formVocabularyAndFeatures(discourseGraphList)
        print(self.vocabDict)
        for class_ in self.vocabDict:
            print(len(self.vocabDict[class_]))

    def fit(self,discourseGraphList):
        """Main process for fitting an M4 to our conversations, represented as
        discourse graphs."""
        self.initializePars(discourseGraphList)
#test process

if __name__ == "__main__":
    #load in discourse graphs
    discourseGraphList = pkl.load(open(
                        "../data/preprocessed/testingDiscourseGraphs.pkl","rb"))
    newM4 = m4(2)
    newM4.fit(discourseGraphList)
