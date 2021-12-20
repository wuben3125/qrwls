import numpy as np
import random as rand
import qiskit
import timeit
import threading
import multiprocessing as mp
import glv

from common import *
from quantum import *

    
def partial_vector(angle_list, gamma, b, c, d_partial, task_number):
    n = len(angle_list)
    
    # for testing
#    print(f"Task Number: {task_number}")
    
    x_estimate_partial = np.array([x_component_qrw(angle_list, gamma, b, x_component_index, c, d_partial) \
                                   for x_component_index in range(2**n)])
    
    return x_estimate_partial
    
def partial_vector_noisy(angle_list, gamma, b, c, d_partial, task_number, backend, test_invalid_steps=True):
    n = len(angle_list)
    
    # for testing
#    print(f"Task Number: {task_number}")
    
    x_estimate_partial = np.zeros(2**n)
    step_counts = np.zeros(3) # valid, invalid, retried
    
    for x_component_index in range(2**n):
        x_component_partial, step_counts_partial = x_component_qrw_noisy(angle_list, gamma, b, x_component_index, c, d_partial, backend, test_invalid_steps=test_invalid_steps)
        x_estimate_partial[x_component_index] = x_component_partial
        step_counts += step_counts_partial
        
    #x_estimate_partial = np.array([x_component_qrw_noisy(angle_list, gamma, b, x_component_index, c, d_partial, backend, test_invalid_steps=test_invalid_steps) \
    #                               for x_component_index in range(2**n)])
    #print()
    
    return x_estimate_partial, step_counts


def qrw_experiment(n, num_samples, angle_lists, b, output_filename, gamma, c, backend, suppress_start_print=False, graph_points=10, min_shots=24, max_shots=1000, test_invalid_steps=True):

#    num_samples = len(angle_lists)
    
    ## output angle lists and gamma
    if not suppress_start_print:
        with open(output_filename, 'a') as file:
            file.write("Angle Lists:")
            file.write('\n')
            
            for angle_list in angle_lists:
                file.write(str(angle_list))
                file.write('\n')
            
            file.write('\n\n')
    
        print(f"gamma = {gamma}")
            
    print(f"cpu_count() = {mp.cpu_count()}")
    
    ## get solution with random walk
    ### number of samples
#    graph_points = 10
    

    for sample_number in np.geomspace(min_shots, max_shots, num = graph_points)[-1::-1]:
    
        start = timeit.default_timer()    
        ### timing
        
        variance = 0 # instantiate
        mean = 0
        standard_error = 0
        std = 0
        
        sample_step_counts = np.zeros((num_samples, 3))
    
        d = int(np.round(sample_number))
        d_24 = int(np.round(d/24))

        print(f"sample_number: {d_24*24}") # for print output
        print()
        with open(output_filename, 'a') as file: # for file output
            file.write(f"sample_number: {d_24 * 24}")
            file.write('\n')
        
        
        error_val = 0
        
        
        for system_idx, angle_list in enumerate(angle_lists[:num_samples]):
        
            # reset global step counters
            #glv.reset_glv()
            
            # get variables
            qrw = QuantumRandomWalkHamming_Noisy(angle_list, backend)
            P = qrw.p_matrix()
            A = np.eye(2**n) - gamma*P
            x = np.linalg.inv(A) @ b
        

            ## get partial components
            processes = []


            ## note: function must be declared before pool is created
            global task
            def task(task_num):
        #         save_partial_vector(angle_list, gamma, b, c, d_24, task_num)
                return partial_vector_noisy(angle_list, gamma, b, c, d_24, task_num, backend, test_invalid_steps=test_invalid_steps)

            
            pool = mp.Pool(mp.cpu_count())
            vals = []


        ### with additional task function
            for task_num in range(24):
                vals.append(task_num)


        ## way of mapping
            results = [pool.apply_async(task, (task_num,)) for task_num in vals]
            [result.wait() for result in results]
            pool.close()
            pool.join()

            print() # newline


            ### use results
            result_tuples = [result.get() for result in results]
            
            partial_vectors = [result_tup[0] for result_tup in result_tuples]
            partial_step_counts = [result_tup[1] for result_tup in result_tuples]
            
            x_estimate = sum(partial_vectors)/len(partial_vectors)
            step_counts = sum(partial_step_counts)
            
            current_error = error(x, x_estimate)
            
            with open(output_filename, 'a') as file:
                file.write(f"system #{system_idx+1} error: {current_error}")
                file.write('\n')
            
            error_val += current_error
            
            variance += current_error**2
            
            sample_step_counts[system_idx, :] = step_counts
            
            # print global step counters
            print(f"system #{system_idx+1}:")
            #print(f"total_valid_steps: {glv.total_valid_steps}")
            #print(f"total_invalid_steps: {glv.total_invalid_steps}")
            #print(f"total_retried_steps: {glv.total_retried_steps}")
            print(f"valid_steps: {step_counts[0]}")
            print(f"invalid_steps: {step_counts[1]}")
            print(f"retried_steps: {step_counts[2]}")

        ############    
        end = timeit.default_timer()
        
        mean_error = error_val / num_samples
        
        variance /= num_samples
        variance -= mean_error**2
        std = variance**0.5
        standard_error = std / num_samples**0.5
        
        avg_step_counts = np.mean(sample_step_counts, axis = 0)
        std_step_counts = np.std(sample_step_counts, axis = 0)
        
        # print total step counts
        print()
        print(f"average step counts: {avg_step_counts}")
        print(f"step count standard deviations: {std_step_counts}")

        print() # print newline
        with open(output_filename, 'a') as file: # file i/o
            file.write("(Overall)")
            file.write(f"Time elapsed: {end-start}")
            file.write('\n')
            file.write(f"Avg error: {mean_error}")
            file.write('\n')
            file.write(f"Standard deviation: {std}")
            file.write('\n')
            file.write(f"Standard error: {standard_error}")
            file.write('\n')
            file.write(f"Avg step counts: {avg_step_counts}")
            file.write('\n')
            file.write(f"Step count standard deviations: {std_step_counts}")
            file.write('\n\n')
