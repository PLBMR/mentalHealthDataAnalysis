#discourseGraph.py
#helper that holds our class for a discourse graph, a tree-like structure that
#represents a conversation

#imports

import nltk
import pandas as pd
import pickle as pkl

#class

class DiscourseGraph:
    def __init__(self):
        self.root = None
        self.title = None #will add to this
    
    def buildChild(self,rootChildID,commentFrame):
        #recursive function for building a child node in the discourse graph
        idStart = 3
        childID = rootChildID[idStart:]
        #get text
        givenText = (
            commentFrame.loc[commentFrame["id"] == childID,"body"].values[0])
        childNode = DiscourseNode(rootChildID,givenText)
        #then build children of child node
        childrenCommentFrame = commentFrame[
                                commentFrame["parent_id"] == rootChildID]
        for cid in childrenCommentFrame["id"]:
            appendedChildID = "t1_" + cid
            childNode.children.append(self.buildChild(appendedChildID,
                                                     commentFrame))
        return childNode

    def build(self,rootPostID,threadFrame,commentFrame):
        #wrapper for building the discourse graph from a given root post
        #get title
        idStart = 3
        titleID = rootPostID[idStart:]
        self.title = (
                threadFrame.loc[threadFrame["id"] == titleID,"title"].values[0])
        #then make root node
        givenText = (
            threadFrame.loc[threadFrame["id"] == titleID,"selftext"].values[0])
        self.root = DiscourseNode(rootPostID,givenText)
        #then find children of root
        childrenCommentFrame = commentFrame[
                                    commentFrame["parent_id"] == rootPostID]
        print(self.root.id_)
        for childID in childrenCommentFrame["id"]:
            appendedChildID = "t1_" + childID
            self.root.children.append(self.buildChild(appendedChildID,
                                                    commentFrame))

class DiscourseNode:
    #holds primarily our text and children
    def __init__(self,givenID,text):
        self.id_ = givenID
        self.string = text
        self.tokens = nltk.word_tokenize(text)
        self.children = [] #will add to this possibly

#testing process

if __name__ == "__main__":
    #get thread dictionary
    threadDict = pkl.load(open("../data/preprocessed/testingThreadDict.pkl","rb"
                            ))
    threadFrame = threadDict["posts"]
    commentFrame = threadDict["comments"]
    #start discourse graph
    discourseGraphList = []
    for i in list(threadFrame.index):
        givenThreadID = threadFrame.loc[i,"id"]
        givenThreadID = "t3_" + givenThreadID #indicates it's a post
        newDiscourseGraph = DiscourseGraph()
        newDiscourseGraph.build(givenThreadID,threadFrame,commentFrame)
        discourseGraphList.append(newDiscourseGraph)
    pkl.dump(discourseGraphList,open(
                        "../data/preprocessed/testingDiscourseGraphs","wb"))
