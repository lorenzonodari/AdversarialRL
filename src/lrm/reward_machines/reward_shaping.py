from collections import defaultdict


class AutomatedRewardShaper:
    """
    Implementation of Automated Reward Shaping for reward machine-based agents.

    While the underlying idea of this approach is highly motivated by the work of [Icarte et al., 2018]
    the actual implementation differs for a few details: the actual potential function that is derived
    is exactly the value function over the reward machine MDP, instead of negative: ie: phi(u) = v(u) instead
    of phi(u) = - v(u).
    """

    def __init__(self, rm_transitions, rm_rewards):

        self._delta_u = rm_transitions
        self._delta_r = rm_rewards

        self._U = {state for state, _ in self._delta_u.keys()} | set(self._delta_u.values())
        self._rm_edges = self._delta_r.keys() | self._delta_u.keys()
        self._relevant_labels = {label for _, label in self._rm_edges}

    def compute_potential_function(self, gamma=0.9):

        # Initialize the value of each RM state
        values = {u: 0 for u in self._U}
        values[None] = 0

        # Value iteration loop
        error = 1
        while error > 0:

            error = 0
            for u in self._U:

                new_values = []
                exiting_labels = {label for state, label in self._rm_edges if state == u}
                for sigma in exiting_labels:

                    try:
                        reward = self._delta_r[u, sigma]
                    except KeyError:
                        reward = 0

                    try:
                        next_state = self._delta_u[u, sigma]
                    except KeyError:
                        next_state = None

                    new_values.append(reward + gamma * values[next_state])

                if len(exiting_labels) > 0:
                    new_best_value = max(new_values)

                    error = max((error, abs(values[u] - new_best_value)))
                    values[u] = new_best_value

        # Return potential function phi(u) = v(u)
        del values[None]
        return values


class NullRewardShaper:
    """
    Convenience class for representing the absence of reward shaping.

    The compute_potential_function() simply returns a dictionary that yields 0 for any requested key.
    """

    def __init__(self):
        self._potential_function = defaultdict(lambda x: 0)

    def compute_potential_function(self, gamma):

        return self._potential_function


