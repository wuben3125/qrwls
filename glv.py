"""
Used for storing global counts of steps to print out in parallel
"""
total_valid_steps = 0
total_invalid_steps = 0
total_retried_steps = 0

def reset_glv():
    total_valid_steps = 0
    total_invalid_steps = 0
    total_retried_steps = 0
