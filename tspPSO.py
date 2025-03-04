import numpy as np
import networkx as nx
import sys

## Class Particle
#  This class is the basic particle class that stores necessary information
#  for each particle in the swarm.
class Particle(object):
    def __init__(self):
        self.pBestCost = None
        self.pBestPos = None
        self.pos = None
        self.vel = None
        self.currentCost = None

## Class SwarmGlobal
# This class stores the global best positions and costs for the entire swarm.  
class SwarmGlobal(object):
    ## Class constructor
    # @param self is the object pointer  
    # @param gBestIntial is the intialization for gBestCost, the global best cost of the swarm
    def __init__(self, gBestInitial):
        self.gBestPos = None
        self.gBestCost = gBestInitial

## Class DiscretePSOLK
# This is the main PSO class that uses a Lin-Kernighan bisection to find new particle positions.
# The Lin-Kernighan bisection optimizes the path by reducing the cost between edges that cross between equal partitions
class DiscretePSOLK:
    ## Class constructor
    # @param self is the object pointer  
    # @param distanceMatrix is an NxN matrix representing the distance between various cities, where N is the number of cities
    # @param cities is a list of the cities, taking the values 0 to number of cities - 1
    # @param rGen is a pseudorandom number generator object used to perform random operations
    def __init__(self, distanceMatrix, cities, rGen):
        self.distanceMatrix = distanceMatrix
        self.cities = cities
        self.rGen = rGen
        # Create a graph representation of the TSP, where each node is a city and the graph is undirected with edge weights corresponding to the distance matrix
        self.tspGraph = self.createTSPGraph()

    ## This method creates a graph of the TSP using weights from the passed distance matrix
    # @returns a undirected graph with edge weights equal to the provided distance matrix
    def createTSPGraph(self):
        # create an adjacency matrix of all ones except for the diagonal
        adj_mat = np.ones(self.distanceMatrix.shape)
        np.fill_diagonal(adj_mat, 0)
        # create the graph
        tspGraph = nx.from_numpy_matrix(adj_mat)

        # fill the graph with weights
        for x,y in tspGraph.edges:
            tspGraph[x][y]['weight'] = self.distanceMatrix[x,y]
            
        return tspGraph
        
    ## This method computes the cost of a path by summing the distances between each city in the path
    # @param path is the path with range 0 to N-1 cities indicating the route from the starting city
    def pathCost(self, path):
        totalDistance = 0
        # sum the distance between each city
        for i in range(len(path)-1):
            totalDistance += self.distanceMatrix[path[i], path[i+1]]
        # return to starting city
        totalDistance += self.distanceMatrix[path[-1], path[0]]

        return totalDistance

    ## This method defines the velocity for the PSO algorithm. Since this is a discrete case, the velocity
    # can take on the values 0, 1, or 2, indication whether to find a new path, follow the particleBest, or follow the global best respectively.
    # The returns a choice from the allowed values based on the probabily definied in probVect.
    # @param probVect This parameter is a list with values {0, 1, 2} representing which direction to move the particle
    def velocityOperator(self, probVect):
        assert sum(probVect) == 1
        return self.rGen.choice(3, p=probVect)

    ## This method defines how the particles new position is assigned. It uses the velocity to either find a new path using the Lin-Kernighan bisection,
    # or go towards global or local bests with path relinking. This method returns the new position.
    # @param velocity This is an integer value representing how to move the particle
    # @param particle The particle object holding the particle data
    # @param swarm The swarm object holding swarm bests
    def positionAssignment(self, velocity, particle, swarm, maxMoves=None):
        # moves its own way
        if velocity == 0:
            newPosition = self.kernighanLinMovement(particle.pos.copy(), maxMoves)
        # moves to pbest
        elif velocity == 1:
            newPosition = self.pathRelinkingForwardBack(particle.pos.copy(), particle.pBestPos.copy(), particle.pBestCost, maxMoves)
        # moves to gbest
        elif velocity == 2:
            newPosition = self.pathRelinkingForwardBack(particle.pos.copy(), swarm.gBestPos.copy(), swarm.gBestCost, maxMoves)
        else:
            # velocity not within specified values
            raise ValueError("Velocity out of bounds: ", velocity)
        
        return newPosition
    
    ## This function shifts a path to the left based on the index of a target value.
    # @param path the path to shift
    # @param targetIndex The index of the target value that is used to shift the path. 
    def shiftLeft(self, path, targetIndex):
        # The targetIndex is negated to shift the path left
        return np.roll(path, -targetIndex)
    
    ## This function creates a path from two splits that indicate moves. The first split indicates the starting city of the move, and the second split indicates the ending city of the move.
    # pairs of moves are alligned on the same index, meaning s1[0] is the starting city and s2[0] is the ending city of the move, and each index represents a move.
    # @param s1 The split containing starting cities of the move
    # @param s2 The split containing ending cities of the move
    def cutCombine(self, s1, s2):
        # returns a list in path format from given moves
        return np.array([x for y in zip(s1, s2) for x in y], dtype=int)
    
    ## This function creates tuple moves for a given path. This can be converted back into a path with cutCombine()
    # @param path The path to split into tuple moves 
    def connectedPathFull(self, path):
        # create two new arrays
        x = np.array([])
        y = np.array([])

        # iterate through the path at a step size of 2 to form tuples
        for i in range(0, len(path), 2):
            # starting city
            x = np.append(x, path[i])
            # ending city
            y = np.append(y, path[i+1])

        return x, y
    
    ## This function moves the particle in a new direction using a Lin-Kernigan bisection. The initial path is cut into tuple moves and 
    # placed back into path format after the operation.
    # @param initialPath The initial path to move
    def kernighanLinMovement(self, initialPath, maxMoves=None):
        # Introduce randomness by changing the initial path randomly in hope of escaping local convergence
        for i in range(self.rGen.choice(len(self.cities))):
            initialPath[i:]= self.shiftLeft(initialPath[i:], self.rGen.choice(len(self.cities)))
        splitIndex = len(initialPath) // 2
        #blocks = nx.community.kernighan_lin_bisection(self.tspGraph, partition=(initialPath[splitIndex:], initialPath[:splitIndex]))
        # create tuple move lists
        x, y = self.connectedPathFull(initialPath)
        # use the kernighan-lin bisection provided by the networkx package to form an optimal cut path
        if maxMoves != None:
            blocks = nx.community.kernighan_lin_bisection(self.tspGraph, partition=(x, y), max_iter=maxMoves)
        else:
            blocks = nx.community.kernighan_lin_bisection(self.tspGraph, partition=(x, y))
        #return np.concatenate((list(blocks[0]),list(blocks[1])))
        # convert the tuple moves back into path format and return
        return self.cutCombine(list(blocks[0]), list(blocks[1]))

    ## This function is called when moving the path to the global or personal best. It shifts each index in the initial path to the left until it is identical to the
    # target path. At each shift, the cost of the path is found to find an optimal path during the shifting process. This function implements a backwards-forwards path relinking,
    # which means that the personalBest moves to the globalBest and vice versa simultaneously, so that there are two path sets being evaluated.
    # @param originalSolution The original solution that needs to be relinked
    # @param targetSolution The target solution that the original solution is relinked to
    # @targetCost the target cost to optimize
    # @param maxMoves an optimal parameter maxMoves to control how many shifts are allowed
    def pathRelinkingForwardBack(self, originalSolution, targetSolution, targetCost, maxMoves=None):
        # set the number of shifts as the length of the target solution or maxMoves if it exists
        n = len(targetSolution) -1 if maxMoves == None else maxMoves
        
        # Create solution placeholders
        optimalSolution = targetSolution
        optimalCost = targetCost
        forwardSolution = originalSolution.copy()
        backwardSolution = targetSolution.copy()

        # for each shift operation
        for i in range(n):
            # forward relink
            targetValue = targetSolution[i]
            splitSequence = forwardSolution[i:]
            rotationValue = np.where(splitSequence == targetValue)[0][0]
            forwardSolution[i:] = self.shiftLeft(splitSequence, rotationValue)
            costForward = self.pathCost(forwardSolution)

            # backward relink
            targetValue = originalSolution[i]
            splitSequence = backwardSolution[i:]
            rotationValue = np.where(splitSequence == targetValue)[0][0]
            backwardSolution[i:] = self.shiftLeft(splitSequence, rotationValue)
            costBackward = self.pathCost(backwardSolution)

            # if the cost of either relink is better than the optimal cost, make it the new optimal
            if costForward < optimalCost:
                optimalSolution = forwardSolution.copy()
                optimalCost = costForward
            
            if costBackward < optimalCost:
                optimalSolution = backwardSolution.copy()
                optimalCost = costBackward
        
        # return the optimal solution    
        return optimalSolution

    ## The run function for the Particle Swarm Optimization algorithm.
    # @param numParticles The number of particles to use for the optimization
    # @param pr1 The probabilty of a move towards a new direction
    # @param pr2 The probabilty of a move towards the particle's personal best
    # @param pr3 The probabilty of a move towards the swarm's global best
    # @param lrP1 The learning rate (or scaler) for the probability of moving in a new direction
    # @param lrP2 The Learning rate (or scaler) for the probability of moving towards a personal best
    # @param maxIterations The maximum number of iterations for the algorithm
    # @param maxMoves The max moves to use for path relinking and Kernighan-Lin
    # @returns The global optimal solution for the swarm
    def runPSO(self, numParticles, pr1, pr2, pr3, lrP1, lrP2, maxIterations, maxMoves=None):
        # probabilites should equal 1
        assert (pr1 + pr2 + pr3) == 1

        # create a list for particles and a swarm object
        particles = []
        swarm = SwarmGlobal(sys.maxsize)

        # initialize the population
        for pIdx in range(numParticles):
            # initalize particle position and costs
            p = Particle()
            p.pos = self.rGen.permutation(self.cities)
            p.pBestCost = self.pathCost(p.pos)
            p.pBestPos = p.pos.copy()
            particles.append(p)

        # Run the optimization
        for k in range(maxIterations):
            # Loop through each particle
            for i in range(numParticles):
                particle = particles[i]
                # get path cost
                pCost = self.pathCost(particle.pos)

                # if the cost is less than personal or global optima, save it and the path to their respective variables
                if pCost < particle.pBestCost:
                    particle.pBestCost = pCost
                    particle.pBestPos = particle.pos.copy()
                if pCost < swarm.gBestCost:
                    swarm.gBestCost = pCost
                    swarm.gBestPos = particle.pos.copy()
                    
                # assign the particle velocity based on the given probabilities
                particle.vel = self.velocityOperator([pr1,pr2,pr3])
                # from the particle velocity, assign a new position
                particle.pos = self.positionAssignment(particle.vel, particle, swarm)

            # Log iteration statistics
            if (k % 50) == 0:
                print(f'Iteration: {k}, Best Cost: {swarm.gBestCost}, probs: {pr1},{pr2},{pr3}')

            # update probabilities
            pr1 = pr1 * lrP1
            pr2 = pr2 * lrP2
            pr3 = 1 - (pr1 + pr2)

        return swarm.gBestPos