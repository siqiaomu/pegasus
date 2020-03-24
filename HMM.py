########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
from numba import jit


@jit(nopython=True)
def forward_static(x, A, O, L, D, A_start):
    
    M = len(x)
    
    alphas = [[0. for _ in range(L)] for _ in range(M + 1)]
    
    for z in range(L):  #iterating through states
        alphas[1][z] = O[z][x[0]] * A_start[z]

    for i in range(1, M): #1, 2, ... M
        for z in range(L):
            aa = 0
            for j in range(L):
                aa +=  alphas[i][j] * A[j][z] 
            alphas[i + 1][z] = aa * O[z][x[i]]

    alphas = np.array(alphas)

    if normalize == True:
        for row in alphas:
            if sum(row) != 0:
                row = row/sum(row)
                
    return alphas


@jit(nopython=True)
def backward_static(x, A, O, L, D, A_start):
    M = len(x)      # Length of sequence.
    betas = [[0. for _ in range(L)] for _ in range(M + 1)]


    for z in range(L):  #iterating through states
        betas[M][z] = 1

    for i1 in range(1, M):
        i = M - i1 
        for z in range(L):
            bb = 0
            for j in range(L):
                bb +=  betas[i + 1][j]*A[z][j]* O[j][x[i]]     
            betas[i][z] = bb 

    betas = np.array(betas)
    if normalize == True:
        for row in betas:
            if sum(row) != 0:
                row = row/sum(row)          

    return betas


@jit(nopython=True)
def unsupervised_static(x, M, alphas, betas, O, A, O_num, O_den, A_num, A_den, L, D):
    for a in range(L):
        for b in range(L):
            for i in range(M - 1):
                P1_num = alphas[i][a] * A[a][b] * O[b][x[i + 1]] * betas[i + 1][b] 

                P1_den = 0
                for aprime in range(L):
                    for bprime in range(L):
                        P1_den += alphas[i][aprime] * A[aprime][bprime] * O[bprime][x[i + 1]] * betas[i + 1][bprime]

                P2_num = alphas[i][a] * betas[i][a]
                P2_den = 0
                for aprime in range(L):
                    P2_den += alphas[i][aprime] * betas[i][aprime]      

                if P1_den != 0:
                    A_num[a][b] += P1_num/P1_den

                if P2_den != 0:
                    A_den[a][b] += P2_num/P2_den

    for z in range(L):
        for w in range(D):
            for i in range(M):

                P3_num = alphas[i][z] * betas[i][z]
                P3_den = 0
                for zprime in range(L):
                    P3_den += alphas[i][zprime] * betas[i][zprime]
                if P3_den != 0:
                    P3 = P3_num/P3_den
                else:
                    P3 = 0


                O_num[z][w] += (x[i] == w) * P3
                O_den[z][w] += P3

    return O_num, O_den, A_num, A_den


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimen sions L x D.
                        The (i, j)^th element is theprobability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of possible observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
#         probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
#         seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        
        A = self.A #The transition matrix
        O = self.O #The observation matrix

        L = self.L #Number of states.
        D = self.D #Number of observations
        
        print('L (the number of states) = ' + str(L))
        
        A_start = self.A_start
        
        
        T1 = np.zeros((L, M))
        T2 = [[0 for _ in range(M)] for _ in range(L)]
        
        X = [0 for _ in range(M)] #the most likely hidden state sequence

        
        for  i in range(L): #iterate through states
            T1[i][0] = A_start[i] * O[i][x[0]]
            T2[i][0] = 0

        for j in range(1, M):
            for i in range(L):
                T_list = []
                
                for k in range(L):
                    T_list.append(T1[k][j - 1] * A[k][i] * O[i][x[j]])
                    
                T1[i][j] = max(T_list)
                T2[i][j] = np.argmax(T_list)
                
        X[M - 1] = np.argmax(T1[:, M - 1])
        for j1 in range(2, M + 1):
            j =M - j1 + 1
            X[j - 1] = T2[X[j]][j]

        return str(X)

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''
        
        x = np.array(x)
        
        A = np.array(self.A) #The transition matrix
        O = np.array(self.O) #The observation matrix

        L = self.L #Number of states.
        D = self.D #Number of observations
        
        A_start = np.array(self.A_start)
        
        alph = forward_static(x, A, O, L, D, A_start)
        
        return alph

    
    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        x = np.array(x)
        
        A = np.array(self.A) #The transition matrix
        O = np.array(self.O) #The observation matrix

        L = self.L #Number of states.
        D = self.D #Number of observations
        
        A_start = np.array(self.A_start)
        
        bets = backward_static(x, A, O, L, D, A_start)
        
        return bets
        
        
    
    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        
        L = max(max(Y)) + 1
        D = max(max(X)) + 1
        
        A_matrix = np.zeros((L, L))
                
        for i in range(L): #also a
            for j in range(L): #also b
                np.array(Y) == i
                
                num = 0
                denom = 0

                for lst in Y: #iterate through list of lists
                    
                    denom += sum(np.array(lst) == i)
                    
                    for k in range(len(lst) - 1): #iterate through indices of items in list 
                        if (lst[k] == i):
                            if lst[k + 1] == j:
                                num += 1

                A_matrix[i][j] = num/denom
                
        self.A = A_matrix

        # Calculate each element of O using the M-step formulas.

        O_matrix = np.zeros((L, D))

        for z in range(L): 
            for w in range(D): 
                denom = 0
                num = 0
                #numerator sum
                for m in range(len(X)): #iterate through indices of lists in list
                    denom += sum(np.array(Y[m]) == z)
                    for n in range(len(X[m])): #iterate through indices of items in list
                        if X[m][n] == w and Y[m][n] == z:
                            num += 1

                O_matrix[z][w] = num/denom
                
        self.O = O_matrix
        

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
     
        L = self.L
        D = self.D
        

#         A = [[random.random() for i in range(L)] for j in range(L)]

#         # Randomly initialize and normalize matrix O.

#         O = [[random.random() for i in range(D)] for j in range(L)]

        
        A = np.array(self.A)
        O = np.array(self.O)
        
        
        for N in range(N_iters):
            A_num = np.zeros((L, L))
            A_den = np.zeros((L, L))
            O_num = np.zeros((L, D))
            O_den = np.zeros((L, D))
            if N % 10 == 0:
                print('Iteration: ' + str(N) + '/' + str(N_iters))
            for g in range(len(X)): #each x[g] is a sequence
                x = np.array(X[g])
                if g % 500 == 0:
                    print('Sequence: ' +  str(g) + '/' + str(len(X)))      
                M = len(x)
                alphas = self.forward(x, normalize=True)[1:]
                betas = self.backward(x, normalize=True)[1:]
                O_num, O_den, A_num, A_den = unsupervised_static(x, M, alphas, betas, O, A, O_num, O_den, A_num, A_den, L, D)
                
            A = np.divide(np.array(A_num), np.array(A_den))
            O = np.divide(np.array(O_num), np.array(O_den))
            
            self.A = A
            self.O = O
            

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        
        O = self.O
        A = self.A
        D = self.D
        L = self.L
        A_start = self.A_start
        
        state0 = np.random.choice(range(len(A_start)), 1, p=A_start)[0]
        emission0 = np.random.choice(range(D), 1, p=O[state0])[0]
        
        states.append(state0)
        emission.append(emission0)
        
        for i in range(1, M):
#             if sum(A[states[i - 1]]) != 1:
#                 print(A[states[i - 1]])
            state = np.random.choice(range(L), 1, p=A[states[i - 1]])[0]
            emiss = np.random.choice(range(D), 1, p=O[state])[0]
            
            states.append(state)
            emission.append(emiss)
        
        return emission, states
    
    def generate_emission2(self, stress_dict, unstress_dict, obs_map):
        '''
        '''
        emission = []
        states = []
        
        O = self.O
        A = self.A
        D = self.D
        L = self.L
        A_start = self.A_start
        
        stress_index = []
        stress_keys = list(stress_dict.keys())
        
        unstress_index = []
        unstress_keys = list(unstress_dict.keys())
        
        for i in range(len(stress_keys)):
            stress_index.append(obs_map[stress_keys[i]])
        
        for j in range(len(unstress_keys)):
            unstress_index.append(obs_map[unstress_keys[j]])
        
        inv_map = {v: k for k, v in obs_map.items()}
        
        counter = 0

        emission0 = '0.001'
        
        while emission0 not in unstress_index:
            state0 = np.random.choice(range(len(A_start)), 1, p=A_start)[0]
            emission0 = np.random.choice(range(D), 1, p=O[state0])[0]
        
        states.append(state0)
        emission.append(emission0)
        
        counter += len(unstress_dict[inv_map[emission0]])
        
        
        while counter < 10:
            state = np.random.choice(range(L), 1, p=A[states[-1]])[0]
            emiss = np.random.choice(range(D), 1, p=O[state])[0]
            if counter % 2 == 0: 
                
                while emiss not in unstress_index:
                    state = np.random.choice(range(L), 1, p=A[states[-1]])[0]
                    emiss = np.random.choice(range(D), 1, p=O[state])[0]
                
                add = len(unstress_dict[inv_map[emiss]])
                
                
            if counter % 2 != 0:
                
                while emiss not in stress_index:
                    state = np.random.choice(range(L), 1, p=A[states[-1]])[0]
                    emiss = np.random.choice(range(D), 1, p=O[state])[0]
                
                add = len(stress_dict[inv_map[emiss]])
              
            
            states.append(state)
            emission.append(emiss)
            counter += add
        
        return emission, states



    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
