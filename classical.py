"""

Code for discrete classical random walk

"""

### imports
import numpy as np
import random as rand
### imports 

def classical_walk(num_nodes, num_steps, coin_bias, num_shots):
    """
    num_nodes: int
    num_steps: int 
    coin_bias: float - value between 0 and 1 that determines probability of moving to the right
    
    performs a 1-D classical random walk on num_nodes nodes, with wraparound
    returns a list of final positions
    """
    
    import random as rand
    
    if coin_bias > 1 or coin_bias < 0:
        raise ValueError("Coin probability in improper range")
        
    ## list of positions to be graphed
    position_list = []
        
    for shot in range(num_shots):
        
        ## start in middle
        position = int( (num_nodes+1)/2 )

        for i in range(num_steps):
            
            ## flip coin
            if rand.random() < coin_bias:
                position = (position+1)% num_nodes
            else:
                position = (position-1)% num_nodes
                
        position_list.append(position)
            
    return position_list


def bit_flip(one_or_zero):
            if one_or_zero == 0:
                return 1
            elif one_or_zero == 1:
                return 0
            else:
                raise Exception("Value not 0 or 1")
                
                
# Random Walk class
class ClassicalRandomWalkHamming:
    
    """
    code for classical random walk (?) 
    """
    
    import numpy as np
    import random as rand
    
    def __init__(self, theta_list):
        
        """
        theta_list: list of angles, whose cos and sin correspond to non-flip and flip probabilities
            [theta_n-1, theta_n-2, ..., theta_1, theta_0]
        
        """
        
        self.theta_list = theta_list
        
        self.n = len(theta_list)
        
    def p_matrix(self):
        """
        generates and returns P matrix denoted by thetas
        
        the J', J-th element is defined by prod_{l=0,...,n-1} cos^2(theta_l/2)^(1-i_l) * sin^2(theta_l/2)^i_l
        
        """
        
        matrix = np.ones((2**self.n, 2**self.n)) # start with one, to be multiplied
        
        for i in range(2**self.n):
            for j in range(2**self.n):
                
                xor_int = i ^ j
                
                xor_binary = int_to_binarylist(xor_int, self.n)
                
                for idx, digit in enumerate(xor_binary):
                    if digit == 1: # flip
                        matrix[i,j] *= np.sin(self.theta_list[idx]/2)**2
                    elif digit == 0:
                        matrix[i,j] *= np.cos(self.theta_list[idx]/2)**2
                    else:
                        raise Exception("Value error")
        
        return matrix 
    
        
    def random_step(self, current_node):
        """
        current_node: list of 0s and 1s [b_n-1, b_n-2, ..., b_1, b_0]
        
        iterates through each digit and flips according to probabilities given by thetas
        
        returns a list of 0s and 1s
        """
        
        def bit_flip(one_or_zero):
            if one_or_zero == 0:
                return 1
            elif one_or_zero == 1:
                return 0
            else:
                raise Exception("Value not 0 or 1")
                
#         for idx, theta in enumerate(self.theta_list):
#             rand_val = rand.random() # simulate coin flip with value between 0 and 1
            
#             if rand_val < (np.sin(theta/2))**2: # flip
#                 current_node[idx] = bit_flip(current_node[idx])
                
#             else: # if rand_val < (np.sin(theta))**2 + (np.cos(theta))**2
#                 pass # don't flip

        for idx, theta in enumerate(self.theta_list): # start with theta_{n-1}, end with theta_0
            rand_val = rand.random() # generate random value between 0 and 1
            
            if rand_val < (np.sin(theta/2))**2:    
                pass
            else: # if rand_val < (np.sin(theta))**2) + np.cos((theta))**2
                current_node[idx] = bit_flip(current_node[idx])
        
        return current_node
        
    def random_walk(self, initial_node, num_steps):
        """
        initial_node: list of 0s and 1s [b_n-1, b_n-2, ..., b_1, b_0]
        
        returns a list of 0s and 1s
        """
        
        current_node = initial_node
        
        for step in range(num_steps):
            current_node = self.random_step(current_node)
            
        return current_node
    
    
def x_component_crw(theta_list, gamma, b, x_component_index, c, d):
    """
    theta_list: list of angles, whose cos and sin correspond to non-flip and flip probabilities
            [theta_n-1, theta_n-2, ..., theta_1, theta_0]
        
        for A = 1 - gamma*P,
        
        A @ x = b
        
        A_inv = sum_{s=0,...,inf} (gamma^s * P^s)
        A_inv_c = sum_{s=0,...,c} (gamma^s * P^s)
        
        x_c = A_inv_c @ b
        
    b: 1D list or np.array representing end result vector
    x_component_index: int
    c: int; cutoff 
    d: int; number of times to perform each step
    
    returns the appropriate index value of x_c (truncated up to error ~ gamma^c)
    """
    crw = ClassicalRandomWalkHamming(theta_list)
    
    prep_node = int_to_binarylist(x_component_index, num_digits = crw.n)
    
    x_component = 0
    
    for s in range(0, c+1): # number of steps starting at 0
        
        total_sum = 0
        
        for iter in range(d):
            final_node = crw.random_walk(prep_node, s)
            final_node_int = binarylist_to_int(final_node)
            total_sum += b[final_node_int]
                    
        avg_sum = total_sum / d
        
        avg_sum *= gamma**s
        
        ## add to component value
        x_component += avg_sum
        
    return x_component