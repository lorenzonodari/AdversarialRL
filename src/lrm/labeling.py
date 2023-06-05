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

        # Substitute the event string and event features vector in step() output
        info['events'] = tampered_events
        info['event_features'] = self.get_event_features(tampered_events)

        return obs, reward, terminated, truncated, info


class RandomLFNoise(LabelTampering):
    """
    Labeling function tamperer that randomly alters each event in the abstract observation with a given probability
    """

    def __init__(self, env, noise_quantity, *, seed=None):

        super().__init__(env)

        assert 0.0 < noise_quantity < 1.0, "Noise quantity must be in range [0, 1]"
        self._noise_quantity = noise_quantity
        self._seed = seed
        self._random = random.Random(self._seed)

    def _tamper_events(self, events):

        tampered_events = ""

        for e in events:

            if self._random.random() > self._noise_quantity:

                substitute_event = e
                while substitute_event == e:
                    substitute_event = self._random.choice(self._all_events)

                assert substitute_event != e, "This should not be happening"
                tampered_events += substitute_event

            else:
                tampered_events += e

        return tampered_events
