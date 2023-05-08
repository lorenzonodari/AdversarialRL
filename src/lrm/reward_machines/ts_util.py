"""
This code evaluates the neighbors of a particular RM in parallel
"""

import multiprocessing
import random
import math


def rm2str(delta, U_max, observations):
    """
    Returns a 'str' if for this reward machine
    """
    rm_str = ""
    for i in range(U_max):
        for o in observations:
            if (i,o) in delta: j = delta[(i,o)]
            else:              j = i
            rm_str += "(%d,%s,%d)"%(i,o,j)
    return rm_str


def evaluate_rm(delta, local_change, tabu_set, U_max, observations, N, traces):
    """
    Returns the cost of this machine given a set of traces (and infinite if the RM is in the tabu list)
    NOTE: local_change is a tuple '(action, ci, cj, co)' which indicates the transition that was 
          changed from the original RM 'delta'. The 'action' can be 'add' or 'rm'
    """
    # Make sure we are not modifying the RM that was passed by the caller
    delta = delta.copy()

    # If any, apply the specified change to the RM
    if local_change is not None:
        action, ci, co, cj = local_change
        if action == "rm":
            # removing this transition
            del delta[(ci, co)]
        elif action == "add":
            # removing 'co' transitions to 'ci'
            for k in range(U_max):
                if (k,co) in delta and ci == delta[(k,co)]:
                    del delta[(k,co)]
            # changing transition '(ci, co)' to point to 'cj'
            delta[(ci, co)] = cj
            # removing 'co' transitions from 'cj'
            if (cj,co) in delta:
                del delta[(cj,co)]
        else:
            assert False, "You shouldn't be here..."

    # Defining the prediction variables
    predictions = {}
    for o in observations:

        # If the N-model for an observation does not contain at least 2 possibile abstract observations, the prediction
        # of L(e_t+1) from L(e_t) is trivial: there is only one possibility
        if len(N[o]) < 2:
            continue

        # Creating the prediction variables
        for i in range(U_max):
            predictions[(i, o)] = set()

    # Simulating a trace
    to_pay = []
    to_remove = set([(i, o) for i, o in delta])  # Intially, we mark every transition for removal
    for trace in traces:
        i = 0  # Reward machine state
        for t in range(1, len(trace)):
            o1, _ = trace[t-1]
            o2, _ = trace[t]

            # Adding the current abstract observation to the prediction set of the previous one
            # If len(N[o1]) == 1 there is no point in adding the prediction, as its contribution to the cost would be
            # log(len(predictions[o1])) == log(1) == 0
            if len(N[o1]) > 1:
                predictions[(i, o1)].add(o2)
                to_pay.append((i, o1))

            # Updating RM state
            if (i, o2) in delta:

                # If we are not at the last timestep in the trace, unmark the transition for
                # removal in the RM. Thus, we only remove transitions in the RM that arise from
                # the last experience in a trace, as it would not lead to any prediction
                if t < len(trace) - 1:
                    to_remove.discard((i, o2))

                # Transition the RM to the state that results from the current state and abstract observation
                i = delta[(i, o2)]

    # Cleaning the RM, removing the transitions that are not useful for prediction
    for k in to_remove:
        del delta[k]

    # In order to check if the RM is in the tabù list, we first convert it to a string representation
    delta_str = rm2str(delta, U_max, observations)
    if delta_str in tabu_set:
        return float('inf'), None, None

    # Computing the cost of this RM
    cost = 0
    for p in to_pay:
        cost += math.log(len(predictions[p]))

    return cost, delta, delta_str


class Worker(multiprocessing.Process):

    def __init__(self, c_id, jobs, delta, tabu_set, U_max, observations, N, traces):
        super().__init__()
        self.c_id = c_id
        self.jobs = jobs
        self.traces = traces
        self.delta = delta
        self.tabu_set = tabu_set
        self.U_max = U_max
        self.observations = observations
        self.N = N
        self.results = multiprocessing.Queue(1)
        self.results.put(None)
    
    def run(self):
        best_cost = float('inf')
        while True:
            # Getting next job
            with self.c_id.get_lock():
                job_id = self.c_id.value 
                self.c_id.value += 1

            if job_id >= len(self.jobs):
                break

            # Evaluating the RM that results from applying the local change "self.jobs[job_id]"
            cost, delta, delta_str = evaluate_rm(self.delta, self.jobs[job_id], self.tabu_set, self.U_max, self.observations, self.N, self.traces)
            if cost < best_cost:
                best_cost = cost
                self.results.get()
                self.results.put((cost, delta, delta_str))
    
    def get_results(self):
        return self.results


def evaluate_neighborhood(n_workers, delta, tabu_set, U_max, observations, N, initial_obs, traces, *, rng=None):

    # Creating the jobs
    jobs = []
    for i in range(U_max):
        for o in observations:
            for j in range(U_max):
                if i == j:
                    continue
                if (i,o) in delta and delta[(i,o)] == j:
                    jobs.append(('rm',i,o,j))
                elif not(i == 0 and o in initial_obs):
                    # we add this transition unless it is the initial state and the observation is in 
                    # the set of initial transitions!
                    jobs.append(('add',i,o,j))

    if rng is None:
        random.shuffle(jobs)
    else:
        rng.shuffle(jobs)

    current_id = multiprocessing.Value('i', 0)
    workers = [Worker(current_id, jobs, delta, tabu_set, U_max, observations, N, traces) for i in range(n_workers)]

    for w in workers:
        w.start()

    best_cost = float('inf')
    best_delta = None
    best_delta_str = None
    for w in workers:
        w.join()
        result = w.get_results().get()
        if result is not None:
            cost, delta, delta_str = result
            if cost < best_cost:
                best_cost = cost
                best_delta = delta
                best_delta_str = delta_str
    return best_cost, best_delta, best_delta_str
