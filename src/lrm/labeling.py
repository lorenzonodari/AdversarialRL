import random
import gymnasium as gym


class Labeling(gym.Wrapper):
    """
    Wrapper that allows to enrich an environment with a Labeling Function.

    A Labeling Function is a function from environment observations and actions to propositional symbols that represent
    high-level events that can be known to have verified in the environment given the last observation received.
    """

    def __init__(self, env, labeling_function_class):

        super().__init__(env)
        self._labeling_function = labeling_function_class
        self._labeling_instance = None
        self._previous_obs = None

    def step(self, action):
        """
        Step the environment and compute the labeling function for the returned observation.

        In order not to interfere with the environment observation space, events generated via the labeling function
        are returned by the step() function via the info dictionary, at the "events" key. Moreover, the "event_features"
        key is also returned, containing a feature-space representation of the generated events, which can be fed as
        input, for instance, to neural network models.

        The original environment is thus required not to return any information at the said keys.

        :param action: The action to take in the environment
        :return: The obs, reward, terminated, truncated, info tuple, with info["events"] containing the labeling
                 function output, ie: info["events"] = labeling_function(obs), and info['event_features'] containing
                 the feature-space representation of the events.
        """

        assert self._labeling_instance is not None, "Environment must be reset before step() can be called"

        obs, reward, terminated, truncated, info = self.env.step(action)
        info['events'] = self._labeling_instance.get_events(self._previous_obs, action, obs)
        info['event_features'] = self._labeling_instance.get_event_features(info['events'])

        # Update internal state
        self._previous_obs = obs

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):

        # Initialize a new instance of LabelingFunction
        self._labeling_instance = self._labeling_function()

        obs, info = self.env.reset(seed=seed, options=options)
        info['events'] = self._labeling_instance.get_events(None, None, obs)
        info['event_features'] = self._labeling_instance.get_event_features(info['events'])

        # Update internal state
        self._previous_obs = obs

        return obs, info

    def close(self):

        self.env.close()

    def get_all_events(self):
        """
        Return a string containing all the possible events"

        :return: A string containing all the possible events
        """

        return self._labeling_instance.get_all_events()

    def get_event_features(self, events):
        """
        Convert an event string to its features-space representation

        :param events: An string of events
        :return: The feature-space representation of the given events
        """

        return self._labeling_instance.get_event_features(events)


class LabelingFunction:
    """
    Base class for labeling functions.

    A labeling function is a function that returns a list of high-level events that occurred in the environment
    given the current observation, the taken action and the resulting observation.

    NB: Since, in general, the task of determining high-level events for an environment might need the labeling
    function to keep track of the state of the environment, each instance of LabelingFunction is assumed to be
    stateful. This means that, for each episode, a new instance must be utilized in order to reset the internal
    state. For this reason, it is strongly reccomended, when creating concrete sub-classes, to keep class initialization
    as light as possible, in order to avoid excessive overheads deriving from class initialization at each episode.

    See the code for the LabeledEnv wrapper for additional insight on expected usage.
    """

    def get_events(self, obs, action, new_obs) -> str:
        """
        Compute the labeling function.

        Note that, for the initial observation - ie: the one returned by env.reset() - this method must be invoked
        as get_events(None, None, initial_obs. Note that all events are assumed to be represented by a single character.
        TODO: Allow bigger event-spaces? Doable in theory, probably impractical due to complexity of RM algorithms

        :param obs: The previous obs obtained from the environment
        :param action: The previous action executed on the environment
        :param new_obs: The observation arising from the execution of action
        :return: A string containing all the events that hold true in the environment
        """
        raise NotImplementedError('Actual labeling functions must inherit from this class and override this method')

    def get_event_features(self, events):
        """
        Convert events strings to feature-space representation.

        This function is used to convert event string - obtained by calling get_events() - to their feature-space
        representation. This is useful when the events are used, for instance, as the input for a neural network model.

        :param events: A string representing the events that hold true in the environment
        :return: The feature-space representation of the given event string
        """
        raise NotImplementedError('Actual labeling functions must inherit from this class and override this method')

    def get_all_events(self):
        """
        Return a string containing all the events that can be generated by this labelling function.

        :return: A string containing all the possible labelling function events
        """
        raise NotImplementedError("Actual labeling functions must inherit from this class and override this method")


class LabelTampering(gym.Wrapper):
    """
    Environment wrapper to allow for labeling function output tampering.

    This wrapper defines a common interface for tampering with the labeling function output of the underlying env.
    Subclasses only need to implement the _tamper_events() method, which should contain the actual tampering logic.
    That method is invoked for every event generated by the labeling function of the underlying environment.
    """

    def __init__(self, env):

        assert hasattr(env, 'get_all_events'), "Wrapped env must be wrapped by Labeling or implement get_all_events()"
        assert hasattr(env, 'get_event_features'), "Wrapped env must be wrapped by Labeling or implement get_event_features()"

        super().__init__(env)

        self._all_events = env.get_all_events()
        self._n_tamperings = 0

    def _tamper_events(self, events):
        """
        Labeling function output tampering logic.

        This method must be implemented by concrete subclasses to implement the actual tampering logic. Note that
        a labeling function tamperer is not required to tamper with every event: based on the case at hand, the output
        of this method can be the same exact event string that was received as input.

        :param events: The event string to be tampered with
        :return: The tampered event string
        """

        raise NotImplementedError('Actual labeling function tamperers must inherit from this class and override this method')

    def step(self, action):
        """
        Step the environment while applying the labeling function tampering procedure

        :param action: The action to be taken by the agent
        :return: obs, reward, terminated, truncarted, info as by Gym APIs
        """

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply the tampering logic
        true_events = info['events']
        tampered_events = self._tamper_events(true_events)

        if tampered_events != true_events:
            self._n_tamperings += 1

        # Substitute the event string and event features vector in step() output
        info['events'] = tampered_events
        info['event_features'] = self.get_event_features(tampered_events)

        return obs, reward, terminated, truncated, info

    @property
    def n_tamperings(self):
        """
        Returns the number of this tamperer has modified the labelling function output so far.

        :return: Number of labelling function output tamperings so far
        """
        return self._n_tamperings


class RandomLFNoise(LabelTampering):
    """
    Labeling function tamperer that randomly alters abstract observation with a given probability
    """

    def __init__(self, env, noise_quantity, *, seed=None):

        super().__init__(env)

        assert 0.0 < noise_quantity < 1.0, "Noise quantity must be in range [0, 1]"
        self._noise_quantity = noise_quantity
        self._seed = seed
        self._random = random.Random(self._seed)

    def _tamper_events(self, events):

        if self._random.random() > self._noise_quantity:

            # Chose a random position in the event string to tamper
            target_index = self._random.randint(0, len(events) - 1)

            # Chose a random substitute event
            substitute = self._random.choice(self._all_events)

            # If the chosen substitute is already present in the true event string, simply remove the original one
            events_list = list(events)
            if substitute in events:
                del events_list[target_index]

            # If not, carry out the substitution
            else:
                events_list[target_index] = substitute

            return "".join(events_list)

        else:

            return events


class EventBlindingAttack(LabelTampering):
    """
    Implementation of the tampering part of the Event Blinding Attack on RM-based agents.

    This tamperer works by removing a specific set of events from the labelling function output, starting
    from the n-th time it appears and reverting to no tampering at all as soon as a different set of event is detected.
    In other words, this tamperer eliminates the n-th occurrence, in the event sequence, of a subsequence of identical,
    consecutive events.

    Alternatively, this tamperer also allows for permanent Event Blinding Attacks, where the target events are always
    removed from the labelling function output, regardless of the number of times they have already appeared.

    NB: In order for the tampering to happen, the labelling function output must contain ALL the target events
    """

    def __init__(self, env, target_events, appearance=None):
        """

        :param env: The environment to be wrapped
        :param target_events: The set of events to be tampered with
        :param appearance: The appearance index (1-based) when the given events must be tampered with, or None.
                           If None, the target_events will always be removed by the labeling function output
        """

        super().__init__(env)

        assert appearance is None or appearance > 0, 'Appearance index must be at least 1, or None'

        self._target = target_events  # Subset of events we want to attack
        self._target_appearance = appearance  # Appearance index we want to tamper

        # Internal state for timed attacks
        self._times_seen = 0  # Number of times we saw our target, not counting consecutive appearances
        self._still_present = False  # True if our target was seen in the previous LF output
        self._tampering = False  # True if we have begun tampering
        self._done = False  # True if the attack has already been carried out

    def _tamper_events_always(self, events):

        return "".join([e for e in events if e not in self._target])

    def _tamper_events_once(self, events):

        # If the attack has already been done, do not tamper
        if self._done:
            return events

        # Check if all the target events are present: if not, we do not tamper this abstract observation
        for e in self._target:

            if e not in events:

                # The target is no longer detected
                self._still_present = False

                # If we had already started tampering, we can now stop as the target is no longer detected.
                if self._tampering:
                    self._tampering = False
                    self._done = True

                return events

        # We detected our target after not detecting it: increment the number of times we saw it
        if not self._still_present:
            self._times_seen += 1
            self._still_present = True

        # If the number of times we saw it matches the requested appearance index, we can tamper
        # Alternatively, if we already started tampering and the target is still detected, continue tampering
        if self._target_appearance == self._times_seen or self._tampering:

            self._tampering = True
            return "".join([e for e in events if e not in self._target])

        # It is not yet time to carry out the attack, do not tamper
        return events

    def _tamper_events(self, events):

        if self._target_appearance is None:

            return self._tamper_events_always(events)

        else:

            return self._tamper_events_once(events)

    def reset(self, **kwargs):

        # Reset internal state
        self._times_seen = 0
        self._done = False
        self._tampering = False
        self._still_present = False

        return self.env.reset(**kwargs)


class EdgeBlindingAttack(LabelTampering):
    """
    Implementation of the tampering part of the Edge-based Blinding Attack on RM-based agents.

    This tamperer works by targeting a set of agent's RM transitions: each time one of the target
    transitions would trigger, the output of the labelling function is modified in order to avoid
    the triggering of the transition.

    This attack can be also be seen as the generalization of the State-based Blinding Attack, where
    the attacker aims at preventing the agent's RM to reach a specific state, as this could be obtained
    by the attacker by simply targeting every transition that would lead to the target state.

    As for the Event-based variation of the Blinding attack, the Event-based Blinding attacks also supports
    two variations: the timed variation, where the targets are attacked only ad a specific appearance index,
    and the persistent one, where the targets are always tampered with.
    """

    def __init__(self, env, agent_rm, target_transitions, appearance=None):
        """
        Initialize the Edge-based Blinding Attack tamperer

        NB: Instances of this class require the agent's RM in order to properly work. This, however, is due to
        the current design of the LabelTrampering._tamper_events() method, which only receives the output of the
        labelling function as an argument: in theory the attack only needs the current agent's RM state. However, since
        passing that information is not allowed by the current API, this workaround is used.

        :param env: The environment to be wrapped
        :param agent_rm: The agent's reward machine
        :param target_transitions: The set of transitions to be tampered with
        :param appearance: The appearance index to be targeted. If None, target every appearance of the targets
        """

        super().__init__(env)

        if not isinstance(target_transitions[0], tuple):
            target_transitions = [target_transitions]

        assert appearance is None or appearance > 0, 'Appearance index must be at least 1, or None'

        self._rm = agent_rm
        self._target_transitions = target_transitions
        self._appearance = appearance

        # Internal RM state
        self._rm_state = None

        # Internal state for timed attacks
        self._times_seen = 0
        self._done = False

    def reset(self, **kwargs):

        # Reset the internal state
        self._rm_state = self._rm.get_initial_state()
        self._times_seen = 0
        self._done = False

        # Reset the underlying environment
        return self.env.reset(**kwargs)

    def _update_rm_state(self, events):

        self._rm_state = self._rm.get_next_state(self._rm_state, events)

    def _tamper_events_once(self, events):

        if self._done:

            self._update_rm_state(events)
            return events

        sorted_events = "".join(sorted(events))

        # Check every target_transition
        for required_state, required_events, _ in self._target_transitions:

            required_events = "".join(sorted(required_events))

            # One of our target transitions is applicable: check if it is time to tamper
            if required_state == self._rm_state and required_events in sorted_events:

                self._times_seen += 1

                # We reached the required appearance index: tamper
                if self._times_seen == self._appearance:

                    # Generate the tampered labelling output
                    tampered_events = sorted_events.replace(required_events, "")

                    # We finally tampered, no need to tamper again in the future
                    self._done = True

                    # Update RM state and return tampered labelling function output
                    self._update_rm_state(tampered_events)
                    return tampered_events

        # None of the target transitions was detected, or it wasn't time to tamper
        self._update_rm_state(events)
        return events

    def _tamper_events_always(self, events: str):

        sorted_events = "".join(sorted(events))

        # Check every target transition
        for required_state, required_events, _ in self._target_transitions:

            required_events = "".join(sorted(required_events))

            # The requirements for this transition are satisfied: tamper
            if required_state == self._rm_state and required_events in sorted_events:

                # Generate tampered labelling output
                tampered_events = sorted_events.replace(required_events, "")

                # Update internal RM state
                self._update_rm_state(tampered_events)
                return tampered_events

        # None of the target transitions was triggered: do not tamper
        self._update_rm_state(events)
        return events

    def _tamper_events(self, events):

        if self._appearance is None:

            return self._tamper_events_always(events)

        else:

            return self._tamper_events_once(events)
