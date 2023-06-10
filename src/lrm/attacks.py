import os
import json
import itertools

from lrm.agents import TrainedLRMAgent
from lrm.labeling import EventBlindingAttack, EdgeBlindingAttack

from testing import get_env_for_agent


def gather_traces(session_name):
    """
    Merge multiple sources of event traces into one single set.

    This function allows one to merge all the transition histories obtained by various calls to
    TrainedLRMAgent.test() into one single set. More specifically, the current interface of this function
    allows one to specify the name of a testing session: all the traces that were produced by testing
    the agents in the given session will be grouped and returned into a single set.

    :param session_name: The name of the testing session to be used as the traces source
    :return: A list containing all the traces from the given session
    """

    all_traces = []

    base_folder = f'results/test/{session_name}'
    for agent_id in [f.name for f in os.scandir(base_folder) if f.is_dir()]:

        traces_file_path = f'{base_folder}/{agent_id}/traces.json'
        with open(traces_file_path, 'r') as traces_file:
            traces = json.load(traces_file)

        all_traces.extend(traces)

    # Convert every list to a tuple to allow for comparisons
    all_traces = [(tuple([tuple(t) for t in h]), tuple(p)) for h, p in all_traces]

    return all_traces


def clean_duplicate_histories(traces):
    """
    Compute the frequency of each unique transition history and clean the traces accordingly.

    Given a list of traces, where each of them is a tuple (transitions, (episode_reward, episode_steps)),
    this function creates a new list of traces where each unique transition sequence appears at most once.
    To do so, the frequency of each of them in the original trace is computed and a new list of traces is returned,
    each now in the form: (unique_transition_sequence, (frequency, average_reward, average_steps))

    :param traces: The list of traces to be processed
    :return: The processed list of traces
    """

    clean_traces = []

    transition_histories = set([t for t, _ in traces])
    for transitions in transition_histories:

        associated_performances = [p for t, p in traces if t == transitions]
        frequency = len(associated_performances)
        avg_reward = sum([p[0] for p in associated_performances]) / frequency
        avg_steps = sum([p[1] for p in associated_performances]) / frequency

        clean_trace = transitions, (frequency, avg_reward, avg_steps)
        clean_traces.append(clean_trace)

    return clean_traces


def sort_traces(traces):
    """
    Sort traces in descending order of frequency and associated agent performance.

    The frequency of each trace's event sequence is used as the primary key, the episodic reward is used as the
    secondary key and the episode steps are used as the tertiary key.

    :param traces: The traces to be sorted
    :return: The same traces, with duplicates removed, sorted in descending order of frequency and agent's performance
    """

    traces = sorted(traces, key=lambda x: x[1][2])  # Episode steps
    traces = sorted(traces, key=lambda x: x[1][1], reverse=True)  # Episode reward
    traces = sorted(traces, key=lambda x: x[1][0], reverse=True)  # Frequency

    return traces


def compress_transition_histories(traces):
    """
    Compress the transition sequences of each trace by eliminating duplicated entries.

    :param traces: The traces to be compressed
    :return: The compressed traces
    """

    compressed_traces = []
    for history, performance in traces:

        compressed_history = []

        current_transition = None
        for transition in history:

            if transition != current_transition:
                compressed_history.append(transition)
                current_transition = transition

        # Re-add performance data to the trace
        compressed_trace = tuple(compressed_history), performance
        compressed_traces.append(compressed_trace)

    return compressed_traces


def find_event_blinding_strategies(traces, *,
                                   use_compound_events=False):
    """
    Given a set of traces, compute the possible options to carry out an Event Blinding Attack.

    An option for a timed Event Blinding Attack is simply a tuple (target_events, appearance_index), where:

    - target_events is a subset of all possible events;
    - appearance_index is the index of the target events appearance that we are targeting

    Alternatively, an appearance_index = None represent the case of a persistent Event Blinding Attack, where
    the target string is always removed form the labelling function output.

    :param traces: The traces to be used to determine potential attack options
    :param use_compound_events: If True, also use the event strings as they appear in the traces as potential targets.
                                If False, use only the atomic events as potential targets.
    :return: Two list of potential attack options ie: [(target1, index1), ..., (targetn, indexn)]
    """

    # Extract the event sequences from the transition history of each trace
    event_sequences = [[e for _, e, _ in h] for h, _ in traces]

    # Determine unique event strings found in the traces
    unique_event_strings = set(itertools.chain(*event_sequences))
    chained_event_strings = "".join(unique_event_strings)
    potential_targets = set(chained_event_strings)  # Get unique characters ie: events

    if use_compound_events:
        potential_targets |= unique_event_strings

    # First, we consider permanent strategies, where we choose to attack every occurrence of the target
    persistent_strategies = [(t, None) for t in potential_targets]

    # Then determine observed appearance indexes for each target to determine potential
    # timed attack strategies
    timed_strategies = []
    for target in potential_targets:

        for sequence in event_sequences:

            appearances = 0
            consecutive = False

            for event_string in sequence:

                if target in event_string and not consecutive:
                    consecutive = True
                    appearances += 1

                    timed_strategies.append((target, appearances))

                else:
                    consecutive = False

    # The previous loop might have inserted the same option multiple times
    # Compute the frequency of each option
    unique_options = set(timed_strategies)
    timed_strategies = [(o, timed_strategies.count(o)) for o in unique_options]

    # Finally, sort the options by descending frequency, then ascending appearance index
    timed_strategies = sorted(timed_strategies, key=lambda x: x[0][1])
    timed_strategies = sorted(timed_strategies, key=lambda x: x[1], reverse=True)
    timed_strategies = [s for s, _ in timed_strategies]  # Discard frequency info

    return timed_strategies, persistent_strategies


def find_edge_blinding_strategies(traces, *, target_states=False):
    """

    Given a set of traces, compute the possible strategies to carry an Edge Blinding Attack.

    Since our benchmark environments lead to perfect RMs that have no loops, the same transition can't happen
    more than once in every episode. For this reason, this method only finds persistent strategies.
    TODO: Generalize to arbitraty environments -> implement timed strategies computation

    :param traces: The traces to be used to determine potential attack strategies
    :param target_states: If True, find strategies for a State Blinding Attack
    :return: A list of potential attack strategies
    """

    transition_histories = [th for th, _ in traces]

    # Determine unique transitions found in traces
    unique_transitions = set(itertools.chain(*transition_histories))

    # Exclude transitions that do not change the RM state
    unique_transitions = [t for t in unique_transitions if t[0] != t[2]]

    # Generic Edge-based attack
    if not target_states:

        potential_targets = [t for t in unique_transitions]

    # State-based attack
    else:

        potential_targets = []
        target_states = {reached_state for _, _, reached_state in unique_transitions}

        for state in target_states:

            entering_transitions = [t for t in unique_transitions if t[2] == state]
            potential_targets.append(entering_transitions)

    # Persistent strategies, where we attack every occurrence of the target transition
    persistent_strategies = [(t, None) for t in potential_targets]

    return persistent_strategies


def rank_event_blinding_strategies(victim_id,
                                   traces, *,
                                   use_compound_events=False,
                                   compression=True,
                                   trials_per_strategy=100,
                                   episode_length=500):

    # Load victim agent and associated environment
    victim = TrainedLRMAgent(victim_id)
    base_env = get_env_for_agent(victim_id)

    # Pre-process the traces
    if compression:
        traces = compress_transition_histories(traces)
    traces = clean_duplicate_histories(traces)
    traces = sort_traces(traces)

    # Compute possible attack strategies
    timed_strats, persistent_strats = find_event_blinding_strategies(traces, use_compound_events=use_compound_events)
    strategies = timed_strats + persistent_strats

    # Now test each strategy in order, to rank them
    scores = {s: None for s in strategies}
    for strat in strategies:

        target, appearance = strat
        env = EventBlindingAttack(base_env, target, appearance)

        _, _, strat_reward, strat_steps, strat_traces = victim.test(env, trials_per_strategy, episode_length)  # TODO: Seeding
        scores[strat] = strat_reward / trials_per_strategy, strat_steps / trials_per_strategy, env.n_tamperings / trials_per_strategy

    # Augment the strategies with their respective scores
    ranked_strategies = [(s, scores[s]) for s in strategies]

    # Sort the ranked_strategies: worse agent performance -> better strategy, more tamperings -> worse strategy
    ranked_strategies = sorted(ranked_strategies, key=lambda s: s[1][2])  # Ascending number of tamperings
    ranked_strategies = sorted(ranked_strategies, key=lambda s: s[1][1], reverse=True)  # Descending number of steps
    ranked_strategies = sorted(ranked_strategies, key=lambda s: s[1][0])  # Ascending reward

    victim.close()

    return ranked_strategies


def rank_edge_blinding_strategies(victim_id,
                                  traces, *,
                                  target_states=False,
                                  compression=True,
                                  trials_per_strategy=100,
                                  episode_length=500):

    victim = TrainedLRMAgent(victim_id)
    base_env = get_env_for_agent(victim_id)

    if compression:
        traces = compress_transition_histories(traces)
    traces = clean_duplicate_histories(traces)
    traces = sort_traces(traces)

    # Compute potential attack strategies
    if not target_states:
        strategies = find_edge_blinding_strategies(traces)
    else:
        strategies = find_edge_blinding_strategies(traces, target_states=True)

    scores = {s: None for s in strategies}
    for strat in strategies:

        target, _ = strat
        env = EdgeBlindingAttack(base_env, victim._rm, target)

        _, _, strat_reward, strat_steps, strat_traces = victim.test(env, trials_per_strategy, episode_length)
        scores[strat] = strat_reward / trials_per_strategy, strat_steps / trials_per_strategy, env.n_tamperings / trials_per_strategy

    # Augment each strategy with its associated score
    ranked_strategies = [(s, scores[s]) for s in strategies]

    # Sort the ranked strategies: worse agent performance -> better strategy, more tamperings -> worse strategy
    ranked_strategies = sorted(ranked_strategies, key=lambda s: s[1][2])  # Ascending number of tamperings
    ranked_strategies = sorted(ranked_strategies, key=lambda s: s[1][1], reverse=True)  # Descending number of steps
    ranked_strategies = sorted(ranked_strategies, key=lambda s: s[1][0])  # Ascending reward

    victim.close()

    return ranked_strategies






