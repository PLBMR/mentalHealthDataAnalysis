#m4.py
"""Contains our implementation of Mixed Membership Markov Models (M4) with their
training and prediction algorithms. To learn more about M4, feel free to look at
http://cmci.colorado.edu/~mpaul/files/emnlp2012-m4.pdf ."""

#imports

import numpy as np
import pandas as pd
from discourseGraph import DiscourseGraph, DiscourseNode
import pickle as pkl

#step size functions

def initStepSizeFunc(t):
    """an initial step size function that comes from othe initial paper listed
    above"""
    initStep = 0.1
    initDivisor = 1000
    return initStep / (initDivisor + t)

#classes

class m4:
    """Class that holds our implementation of Mixed Membership Markov Models
    (M4)."""

    def __init__(self,numClasses,stepSizeFunc,transitionWeightVar = 1,
                 dirichletScalePar = 1,tol = .001):
        self.numClasses = numClasses
        self.stepSizeFunc = stepSizeFunc
        self.t = 0 #for the step size function
        self.transitionWeightVar = transitionWeightVar
        self.dirichletScalePar = dirichletScalePar
        #initialize class transition matrix
        self.classTransitionMat = np.random.normal(loc = 0,
                                      scale = np.sqrt(self.transitionWeightVar),
                                      size = (numClasses,numClasses + 2))
        #stopping parameter
        self.tol = tol

    def initDiscourse(self,discourseNode):
        """Helper for initializing the feature vector of a given discourse node
        and then forming the vocabulary present from the discourse succeeding
        said discourse node"""
        #create feature vector
        discourseNode.featureVec = np.zeros(self.numClasses + 2)
        discourseNode.featureVec[self.numClasses + 1] = 1 #bias feature
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
            childVocab = self.initDiscourse(child)
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
            givenDiscourseVocab = self.initDiscourse(discourseGraph.root)
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
    
    def makeExpVec(self,featureVec):
        """helper for making the soft-max vector for the transition from a
        parent's feature vector"""
        expVec = np.exp(self.classTransitionMat.dot(featureVec.reshape(-1,1)))
        expVec = expVec / expVec.sum() #normalize
        #flatten
        expVec = np.squeeze(expVec)
        return expVec
    
    def assignTopicsToDiscourse(self,discourseNode,discourseNodeParent = None):
        """main process for assigning latent class assignment on a given
        discourse node and then recursing through the children of the
        discourseNode"""
        #get parent contribution
        givenDist = 1
        if (type(discourseNodeParent) != type(None)):
            givenDist *= self.makeExpVec(discourseNodeParent.featureVec)
        else:
            parentVec = [0 for class_ in range(self.numClasses)]
            parentVec.append(1) #for start
            parentVec.append(1) #for bias
            givenDist *= self.makeExpVec(np.array(parentVec)) #go back to 
                                                            #fixing this later
        #then get child contribution
        childContrib = 1
        for child in discourseNode.children:
            #get particular child contribution
            childExpVec = self.makeExpVec(discourseNode.featureVec)
            for i in range(childExpVec.shape[0]):
                childExpVec[i] **= child.featureVec[i]
            givenChildContrib = 1
            for i in range(childExpVec.shape[0]):
                givenChildContrib *= childExpVec[i]
            #then multiply it into the main contribution
            childContrib *= givenChildContrib
        #then renormalize child contribution in case of integer underflow
        givenDist *= childContrib
        #then normalize
        if (np.sum(givenDist) == 0.0): #just make flat prior
            for i in range(givenDist.shape[0]):
                givenDist[i] = 1 / float(givenDist.shape[0])
        else:
            givenDist /= np.sum(givenDist).astype("float")
        #then move through our words
        for i in range(len(discourseNode.tokens)):
            #get word multiplier
            word = discourseNode.tokens[i]
            nwk = np.array([self.vocabDict[class_][word] for class_ in
                            range(self.numClasses)])
            nk = np.array([sum(self.vocabDict[class_].values()) for class_ in
                            range(self.numClasses)])
            vocabSize = len(self.vocabDict[0])
            wordMul = (nwk + self.dirichletScalePar).astype("float")/(
                        nk + vocabSize * self.dirichletScalePar)
            #then get new distribution
            newDist = givenDist * wordMul
            newDist /= np.sum(newDist)
            #then sample a new class:
            prevClass = discourseNode.topicAssignment[i]
            newClass = np.random.choice(list(range(self.numClasses)),
                                        size = 1,p = newDist)[0]
            discourseNode.topicAssignment[i] = newClass
            #then perform reassignment in other terms
            discourseNode.featureVec[prevClass] -= 1
            discourseNode.featureVec[newClass] += 1
            self.vocabDict[prevClass][word] -= 1
            self.vocabDict[newClass][word] += 1
        #then do this for each child
        for child in discourseNode.children:
            self.assignTopicsToDiscourse(child,discourseNode)

    def eStep(self,discourseGraphList):
        """helper for assigning our latent class assignments on each discourse
        """
        for discourseGraph in discourseGraphList:
            self.assignTopicsToDiscourse(discourseGraph.root)

    def getPartialForNode(self,givenClass,givenComp,discourseNode,
                          parentNode = None):
        """helper for getting the partial derivative associated with a
        particular discourse node"""
        #get parent vector
        if (type(parentNode) == type(None)): #start indicator
            parentVec = [0 for class_ in range(self.numClasses)]
            parentVec.append(1) #for start
            parentVec.append(1) #for bias
            parentVec = np.array(parentVec)
        else:
            parentVec = parentNode.featureVec
        #get our parameters
        ak = parentVec[givenComp]
        nzb = discourseNode.featureVec[givenClass] #number of times class z 
                                                    #occurs in block b
        nb = np.sum(discourseNode.featureVec[0:self.numClasses])
        expVecComp = self.makeExpVec(parentVec)[givenClass]
        #then calculate partial
        partialContrib = ak * (nzb - nb * expVecComp)
        #then add partial contributions from children
        for child in discourseNode.children:
            partialContrib += self.getPartialForNode(givenClass,givenComp,
                                                     child,discourseNode)
        return partialContrib

    def mStep(self,discourseGraphList):
        """helper for maximizing the likelihood of generating our corpus"""
        tol = 0
        for class_ in range(self.numClasses):
            for comp in range(self.classTransitionMat.shape[1]):
                #initialize partial derivative
                partialDeriv = (-1 * self.classTransitionMat[class_,comp]
                                / self.transitionWeightVar)
                for discourseGraph in discourseGraphList:
                    partialDeriv += self.getPartialForNode(class_,comp,
                                                      discourseGraph.root)
                #then update our parameter with our partial derivative
                stepSize = self.stepSizeFunc(self.t)
                self.classTransitionMat[class_,comp] = (
                                            self.classTransitionMat[class_,comp] 
                                            + stepSize * partialDeriv)
                tol += np.abs(stepSize * partialDeriv)
        self.t += 1
        return tol

    def fit(self,discourseGraphList):
        """Main process for fitting an M4 to our conversations, represented as
        discourse graphs."""
        self.initializePars(discourseGraphList)
        #then run through steps
        givenTol = 1
        while (givenTol >= self.tol):
            self.eStep(discourseGraphList)
            givenTol = self.mStep(discourseGraphList)
            print(givenTol)
        print(self.t)
        print(self.classTransitionMat)
#test process

if __name__ == "__main__":
    #load in discourse graphs
    discourseGraphList = pkl.load(open(
                        "../data/preprocessed/testingDiscourseGraphs.pkl","rb"))
    newM4 = m4(3,initStepSizeFunc)
    newM4.fit(discourseGraphList)
