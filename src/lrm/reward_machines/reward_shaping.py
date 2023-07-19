class AutomatedRewardShaper:

    def __init__(self, rm_transitions, rm_rewards):

        self._delta_u = rm_transitions
        self._delta_r = rm_rewards

        self._relevant_labels = {label for _, label in self._delta_u.keys()}
        self._U = {state for state, _ in self._delta_u.keys()} | set(self._delta_u.values())

    def compute_potential_function(self, gamma=0.9):

        # Initialize the value of each RM state
        values = {u: 0 for u in self._U}

        # Value iteration loop
        error = 1
        while error > 0:

            error = 0
            for u in self._U:

                new_values = []
                exiting_labels = {label for state, label in self._delta_u.keys() if state == u}
                for sigma in exiting_labels:

                    try:
                        reward = self._delta_r[u, sigma]
                    except KeyError:
                        reward = 0

                    new_values.append(reward + gamma * values[self._delta_u[u, sigma]])

                if len(exiting_labels) > 0:
                    new_best_value = max(new_values)

                    error = max((error, abs(values[u] - new_best_value)))
                    values[u] = new_best_value

        # Return potential function phi(u) = -v(u)
        return {u: -values[u] for u in self._U}

