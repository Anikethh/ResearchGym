import time
import logging
logger = logging.getLogger("ml-master")

def linear_decay(t, initial_C=1.414, alpha=0.01, lower_bound=0.7):
    '''
    linear decay
    '''
    return max(initial_C-alpha*t, lower_bound)

def exponential_decay(t, initial_C=1.414, gamma=0.99, lower_bound=0.7):

    return max(initial_C * (gamma ** t), lower_bound)

def piecewise_decay(t, initial_C=1.414, T1=100, T2=200, alpha=0.01, lower_bound=0.7):
    if t < T1:
        return initial_C
    elif T1 <= t <= T2:
        return max(initial_C - alpha * (t - T1), lower_bound)
    else:
        return lower_bound

def dynamic_piecewise_decay(steps_limit, n_nodes, initial_C, start_time, time_limit, alpha=0.01, lower_bound=0.7, phase_ratios=[0.3, 0.7]):
    '''
    Dynamic segmented decay strategy: adjust exploration parameter C based on remaining time and the number of nodes already generated
    Args:
    '''
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(time_limit - elapsed_time, 1e-5)

    if elapsed_time > 0:
        generation_speed = n_nodes / elapsed_time
    else:
        generation_speed = 1
    
    logger.info(f'The generation speed of node is {generation_speed} n/s.')
    
    n_remaining = round(generation_speed * remaining_time)
    N_est = n_nodes + n_remaining
    N_est = min(N_est, steps_limit)
    logger.info(f"The estimated total number of nodes is {N_est}")

    progress = n_nodes / N_est if N_est > 0 else 0
    logger.info(f"Based on the estimated total number of nodes, the current progress is {progress}.")

    phase1_end = phase_ratios[0]
    phase2_end = phase_ratios[1]

    if progress < phase1_end:
        return initial_C
    elif progress < phase2_end:
        decay_length = phase2_end - phase1_end
        decay_progress = (progress - phase1_end) / decay_length
        C = initial_C - alpha * decay_progress * N_est
        return max(C, lower_bound)
    else:
        return lower_bound



    
