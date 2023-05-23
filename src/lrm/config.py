import configparser


class LRMConfig(dict):
    def __init__(self, config_file=None, **kwargs):

        super().__init__()

        # Initialize default value for every allowed configuration parameter
        self._init_defaults()

        # Load configuration from config file, if specified
        if config_file is not None:
            self._load_config(config_file)

        # Update the value for explicitly specified parameters
        for key, value in kwargs.items():

            try:
                current_value = self[key]

                # Type checking
                if type(current_value) is not type(value):
                    raise ValueError(f'Invalid type for "{key}": got {type(value)}, need {type(current_value)}')

                setattr(self, key, value)

            except KeyError as exception:
                raise ValueError(f'Unknown configuration parameter: "{key}"') from exception

    def _init_defaults(self):

        # Reward Machines configuration
        self["test_freq"] = int(1e4)  # Frequency of agent's performance testing - in number of learning steps
        self["rm_init_steps"] = int(200e3)  # Number of initial warmup steps to run before learning the RM
        self["rm_u_max"] = 10  # Max number of states that the RM is assumed to have
        self["rm_preprocess"] = True  # If True, preprocess the traces before using them to learn the RM
        self["rm_tabu_size"] = int(1e4)  # Tabu list size
        self["rm_lr_steps"] = 100  # Number of learning steps for Tabu Search
        self["rm_workers"] = 10  # Number of worker threads for Tabu Search
        self["use_perfect_rm"] = False  # If True, use the handcrafted perfect RM instead of learning it

        # Generic RL configuration
        self["gamma"] = 0.9  # Discount factor - value in [0, 1]
        self["train_steps"] = int(5e5)  # Total number of training steps to execute
        self["episode_horizon"] = int(5e3)  # Max allowed episode lenght - longer episodes are truncated
        self["epsilon"] = 0.1  # Epsilon value for eps-greedy exploration strategy - value in [0,1]
        self["max_learning_steps"] = int(5e5)  # Max number of learning steps for a single policy

        # Deep Q-Network configuration
        self["lr"] = 5e-5  # Learning rate
        self["learning_starts"] = int(5e4)  # Number of initial iterations before starting the DQN learning
        self["train_freq"] = 1  # Frequency of DQN learning - in number of agent experiences
        self["target_network_update_freq"] = 100  # Frequency of target network weight update
        self["buffer_size"] = int(1e5)  #
        self["batch_size"] = 32  # Batch size for the DQN
        self["use_double_dqn"] = True  # If True, use the Double-DQN variant for learning the policies
        self["num_hidden_layers"] = 5  # Number of hidden layers of the DQNs
        self["num_neurons"] = 64  # Number of neurons in each hidden layer of the DQNs
        self["use_qrm"] = True  # If True, use the QRM algorithm to train the DQNs

        # Prioritized Experience Replay configuration
        self["prioritized_replay"] = False  # If True, use Prioritized Experience Replay
        self["prioritized_replay_alpha"] = 0.6  # Alpha parameter
        self["prioritized_replay_beta0"] = 0.4  # Initial value for beta parameter
        self["prioritized_replay_beta_iters"] = None  # Number of iterations for each beta value
        self["prioritized_replay_eps"] = 1e-6  # Epsilon parameter

    def _load_config(self, config_file):

        # TODO: Implement
        pass
