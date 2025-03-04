import numpy as np
from collections import deque
import gzip
import tsplib95

## @package tspUtilities
# A simple helper file containing functions to read files proved by the TSPLIB

## This method reads the TSP files from TSPLIB95, the benchmark TSP problems
# @param filePath The file path of the TSP file
# @returns The distnace matrix and cities of the TSP problem
def readATSPFileMatrix(filePath):
    # read the file into a dictionary (following json-like format)
    fileDict = {}
    lastKeyword = None

    # open the file
    with gzip.open(filePath, 'r') as f:
        for line in f:
            # save each word or datapoint into a queue
            words   = deque(line.decode('utf-8').split())
            # get the field keyword from the start of the queue
            keyword = words[0].strip(": ")
            # if the keyword is not numeric, pop it from the queue
            if not words[0].isnumeric():
                words.popleft()

            # if the keyword is numeric, then it is part of the distance matrix
            if keyword.isnumeric():
                # add the values of the distance matrix to the dictionary
                fileDict[lastKeyword] += ','.join(list(words)) + ','

            # if the keyword is not numeric, it is a key in the dictionary
            elif not keyword.isnumeric():
                fileDict[keyword] = ' '.join(list(words))

                lastKeyword = keyword

    # get the dimension and distance matrix
    dimension = int(fileDict['DIMENSION'])
    currentIndex = 0
    processString = [x for x in fileDict['EDGE_WEIGHT_SECTION'].split(',') if x != '']
    outMatrix = np.zeros((dimension,dimension), dtype='int')

    # create a numpy matrix from the distance matrix string
    for i in range(0,dimension):
        for j in range(0, dimension):
            outMatrix[i, j] = processString[currentIndex]
            currentIndex += 1
    fileDict['EDGE_WEIGHT_SECTION'] = outMatrix

    # get the list of cities
    cities = np.arange(dimension, dtype=int)
    
    # return the matrix and the cities
    return outMatrix, cities

## This method reads euclidean STSP files and returns their respective distnace matrix
# @param filePath The path to the TSP file
# @returns The distnace matrix and cities of the TSP problem
def readSTSPFileMatrix(filePath):
    # open the TSP file
    with gzip.open(filePath) as f:
        # read the file contents
        text = f.read().decode('utf-8')

    # use the TSPLIB95 package to parse the file contents
    stspProblem = tsplib95.parse(text)

    # create an empty distane matrix and fill it with the euclidean distances from the TSP problem
    distanceMatrix = np.zeros((stspProblem.dimension, stspProblem.dimension))
    for i in range(1, stspProblem.dimension+1):
        for k in range(1, stspProblem.dimension+1):
            distanceMatrix[i-1,k-1] = stspProblem.get_weight(i,k)

    # get the cities as a numpy array
    cities = np.arange(stspProblem.dimension, dtype=int)
    # return the distance matrix and the cities
    return distanceMatrix, cities