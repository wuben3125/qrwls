"""

Code for both discrete classical random walks and discrete quantum random walks

"""

### imports
import numpy as np
import random as rand
### imports

## code for plotting discrete random walks


## code for solving linear systems with random walks
class RandomWalk:
    
    ###
    # A class that implements a random walk,
    # constructed as a stochastic matrix of transition probabilities
    
    # The walk is constructed with a step-by-step simulation approach,
    # rather than a distribution-oriented approach
    #
    ###
    
    import numpy as np
    import random as rand
    
    def __init__(self, stochastic_matrix):
        """
        stochastic_matrix: a symmetric stochastic numpy array
        
        """
        
        self.num_nodes = len(stochastic_matrix)
        
        self.probabilities = stochastic_matrix
        
        ###
        ## room for type checking (unnecessary)
        ###
    
    def random_step(self, current_node):
        """
        current_node: int from 0 to self.num_nodes-1
        
        returns the next randomly stepped node, according to self.probabilities
        """
        
        probability_row = self.probabilities[current_node]
                
        probability_value = 0
        
        probability_rng = rand.random() # generates a float from 0 to 1
        
        for next_node in range(self.num_nodes):
            ## partitions values 0-1 according to transition probabilities
            probability_value += probability_row[next_node]
                        
            if probability_rng <= probability_value: 
                break
                
        return next_node
        
    
    def walk(self, initial_node, num_steps):
        """
        initial_node: int from 0 to self.num_nodes - 1
        
        num_steps: int
        
        returns the node number of the final node
        """
        
        current_node = initial_node
        
        for step in range(num_steps):
            
            current_node = self.random_step(current_node)
            
        return current_node
            
        
def x_component_mc(P, gamma, b, x_component_index, c, d):
    """
    P: NxN stochastic numpy array for transition probabilities
    gamma: double, such that
        
        for A = 1 - gamma*P,
        
        A @ x = b
        
        A_inv = sum_{s=0,...,inf} (gamma^s * P^s)
        A_inv_c = sum_{s=0,...,c} (gamma^s * P^s)
        
        x_c = A_inv_c @ b
        
    b: list representing end result vector
    x_component_index: int
    c: int
    d: int; number of times to perform each step
    
    returns the appropriate index value of x_c (truncated up to error ~ gamma^c)
    """
    randomwalk = RandomWalk(P)
    
    prep_node = x_component_index
    
    x_component = 0
    
    for s in range(0, c+1): # number of steps starting at 0
        
        total_sum = 0
        
        for iter in range(d):
            final_node = randomwalk.walk(prep_node, s)
            total_sum += b[final_node]
                    
        avg_sum = total_sum / d
        
        avg_sum *= gamma**s
        
        ## add to component value
        x_component += avg_sum
        
    return x_component



def binarylist_to_int(binary_list):
    
    sum = 0
    
    two_power = 1
    
    for digit in binary_list[-1::-1]:
        sum += two_power * digit
        
        two_power *= 2
    
    return sum



def int_to_binarylist(integer, num_digits):
    """
    integer: int
    
    num_digits: int, at least 1
    
    returns a list of length num_digits, including leading 0s if necessary
    """
    
    if num_digits < len(bin(integer)) - 2:
        raise Exception("num_digits too short")

    
    binary_list = []
    
    for i in range(num_digits)[::-1]:
        two_power = 2**(i)
        
        binary_list.append(int(integer/two_power))
        
        integer -= int(integer/two_power) * two_power
        
    return binary_list


def error(x, x_estimate):
    return np.linalg.norm(x-x_estimate) / np.linalg.norm(x)