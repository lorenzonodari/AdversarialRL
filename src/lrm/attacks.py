import os
import json
import itertools

from lrm.agents import TrainedLRMAgent
from lrm.labeling import EventBlindingAttack

from testing import get_env_for_agent


def gather_traces(session_name):
    """
    Merge multiple sources of event traces into one single set.

    This function allows one to merge all the event traces obtained by various calls to
    TrainedLRMAgent.test() into one single set. More specifically, the current interface of this function
    allows one to specify the name of a testing session: all the traces that were produced by testing
    the agents in the given session will be grouped and returned into a single set.

    :param session_name: The name of the testing session to be used as the traces source
    :return: A list containing all the event traces from the given session
    """

    all_traces = []

    base_folder = f'results/test/{session_name}'
    for agent_id in [f.name for f in os.scandir(base_folder) if f.is_dir()]:

        traces_file_path = f'{base_folder}/{agent_id}/traces.json'
        with open(traces_file_path, 'r') as traces_file:
            traces = json.load(traces_file)

        all_traces.extend(traces)

    # Convert each trace element from lists to tuples to allow for comparison
    all_traces = [(tuple(t[0]), tuple(t[1])) for t in all_traces]
    return all_traces


def clean_duplicate_event_sequences(traces):
    """
    Compute the frequency of each unique event sequence and clean the traces accordingly.

    Given a list of traces, where each of them is a tuple (event_sequence, (episode_reward, episode_steps)),
    this function creates a new list of traces where each unique event_sequence appears at most once. To do so,
    the frequency of each event sequence in the original trace is computed and a new list of traces is returned,
    each now in the form: (unique_event_sequence, (frequency, average_reward, average_steps))

    :param traces: The list of traces to be processed
    :return: The processed list of traces
    """

    clean_traces = []

    evt_sequences = set([t[0] for t in traces])
    for evt_seq in evt_sequences:

        associated_performances = [t[1] for t in traces if t[0] == evt_seq]
        frequency = len(associated_performances)
        avg_reward = sum([p[0] for p in associated_performances]) / frequency
        avg_steps = sum([p[1] for p in associated_performances]) / frequency

        clean_trace = evt_seq, (frequency, avg_reward, avg_steps)
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


def compress_event_sequences(traces):
    """
    Compress the event sequence of each trace by eliminating repeated consecutive entries.

    This function aims at simplifying the event sequence contained in each trace by compressing it as follows:

        e.g: 'a', 'a', 'a', 'b', 'b', 'a -> 'a', 'b', 'a'

    More specifically, repeated events are simply kept once every time they appear.

    :param traces: The traces to be compressed
    :return: The compressed traces
    """

    compressed_traces = []
    for trace in traces:

        compressed_evt_sequence = []

        evt_sequence = trace[0]
        current_evt = None

        # Compress the event sequence
        for evt in evt_sequence:

            if evt != current_evt:
                compressed_evt_sequence.append(evt)
                current_evt = evt

        # Re-add performance data to the trace
        compressed_trace = tuple(compressed_evt_sequence), trace[1]
        compressed_traces.append(compressed_trace)

    return compressed_traces


def find_event_blinding_strategies(traces, *,
                                   use_compound_events=False):
    """
    Given a set of traces, compute the possible options to carry out an Event Blinding Attack.

    An option for an Event Blinding Attack is simply a tuple (target_events, appearance_index), where:

    - target_events is a subset of all possible events;
    - appearance_index is the index of the target events appearance that we are targeting

    Alternatively, an appearance_index = None represent the case of a permanent Event Blinding Attack, where
    the target string is always removed form the labelling function output.

    :param traces: The traces to be used to determine potential attack options
    :param use_compound_events: If True, use the event strings as they appear in the traces as potential targets.
                                If False, use atomic events as potential targets.
    :return: Two list of potential attack options ie: [(target1, index1), ..., (targetn, indexn)]
    """

    # Determine unique event strings found in the traces
    event_sequences = [t[0] for t in traces]
    unique_event_strings = set(itertools.chain(*event_sequences))

    if not use_compound_events:

        chained_event_strings = "".join(unique_event_strings)
        potential_targets = set(chained_event_strings)  # Get unique characters ie: events

    else:
        potential_targets = unique_event_strings

    # First, we consider permanent strategies, where we choose to attack every occurrence of the target
    permanent_strategies = [(t, None) for t in potential_targets]

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

    return timed_strategies, permanent_strategies


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
        traces = compress_event_sequences(traces)
    traces = clean_duplicate_event_sequences(traces)
    traces = sort_traces(traces)

    # Compute possible attack strategies
    timed_strats, permanent_strats = find_event_blinding_strategies(traces, use_compound_events=use_compound_events)
    strategies = timed_strats + permanent_strats

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

    return ranked_strategies






