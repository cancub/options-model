import numpy as np
import os
import random

class ReplayBuffer(object):
    '''
    Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
    '''

    def __init__(
        self,
        input_length,
        size=1000000,
        history_length=4,
        use_per=True
    ):
        '''
        Arguments:
            size: Integer, Number of stored transitions

            input_length: Shape of the preprocessed frame

            history_length: Integer, Number of frames stacked together to
                            create a state for the agent

            use_per: Use PER instead of classic experience replay
        '''
        self.size = size
        self.input_length = input_length
        self.history_length = history_length

        # Total index of memory written to, always less than self.size.
        self.count = 0

        # Index to write to.
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.timepoints = np.empty(
            (self.size, input_length * history_length),
            dtype=np.float32
        )
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def add_experience(
        self,
        action,
        timepoint,
        reward,
        terminal,
        clip_reward=True
    ):
        '''
        Saves a transition to the replay buffer

        Arguments:
            action: An integer between 0 and env.action_space.n - 1 
                    determining the action the agent perfomed

            timepoint: A strategy timepoint

            reward: A float determining the reward the agend received for
                    performing an action

            terminal: A bool stating whether the episode terminated
        '''
        if timepoint.shape != self.input_length:
            raise ValueError('Dimension of timepoint is wrong!')

        if clip_reward:
            reward = np.clip(reward, -1.0, 1.0)

        # Write memory
        self.actions[self.current] = action
        self.timepoints[self.current, ...] = timepoint
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        # make the most recent experience important
        self.priorities[self.current] = max(self.priorities.max(), 1)
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.7):
        '''
        Returns a minibatch of self.batch_size = 32 transitions

        Arguments:
            batch_size: How many samples to return

            priority_scale: How much to weight priorities.
                            0 = completely random,
                            1 = completely based on priority

        Returns:
            A tuple of states, actions, rewards, new_states, and terminals

            If use_per is True:
                An array describing the importance of transition. Used for
                scaling gradient steps.

                An array of each index that was sampled.
        '''

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # Get sampling probabilities from priority list
        if self.use_per:
            # We want to return the full history sequence, so we need to start
            # a couple steps down the road (self.history_length).
            scaled_priorities = self.priorities[
                self.history_length:self.count -1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices.
        indices = []
        for _ in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame
                # written with probabilities based on priority weights.
                if self.use_per:
                    end_index = np.random.choice(
                        np.arange(self.history_length, self.count-1),
                        p=sample_probabilities
                    )
                else:
                    end_index = random.randint(
                        self.history_length,
                        self.count - 1
                    )

                # We check that all frames are from same episode with the two
                # following if statements.  If either are True, the index is
                # invalid.
                start_index = end_index - self.history_length

                # Straddling the update index.
                if (start_index <= self.current
                        and end_index >= self.current):
                    continue

                # Only the last index may be terminal.
                if self.terminal_flags[start_index:end_index].any():
                    continue

                break

            indices.append(end_index)

        # Retrieve states from memory.
        states = []
        new_states = []
        for end_idx in indices:
            start_idx = end_idx - self.history_length
            # How it started.
            states.append(
                self.timepoints[start_idx : end_idx , :]
            )

            # How it's going.
            new_states.append(
                self.timepoints[start_idx+1 : end_idx+1 , :]
            )

        states = np.transpose(np.asarray(states), axes=(0, 2, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 1))

        if self.use_per:
            # TODO: read https://arxiv.org/pdf/1511.05952.pdf
            # Get importance weights from probabilities calculated earlier.
            start_probs = sample_probabilities[
                [index - self.history_length for index in indices]
            ]
            importance = 1 / (self.count * start_probs)
            importance = importance / importance.max()

            return (
                (
                    states,
                    self.actions[indices],
                    self.rewards[indices],
                    new_states,
                    self.terminal_flags[indices]
                ),
                importance,
                indices
            )
        else:
            return (
                states,
                self.actions[indices],
                self.rewards[indices],
                new_states,
                self.terminal_flags[indices]
            )

    def set_priorities(self, indices, errors, offset=0.1):
        '''
        Update priorities for PER

        Arguments:
            indices: Indices to update

            errors: For each index, the error between the target Q-vals and the
                    predicted Q-vals
        '''
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        '''Save the replay buffer to a folder'''

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(os.path.join(folder_name,'actions.npy'), self.actions)
        np.save(os.path.join(folder_name,'frames.npy'), self.timepoints)
        np.save(os.path.join(folder_name,'rewards.npy'), self.rewards)
        np.save(
            os.path.join(folder_name,'terminal_flags.npy'),
            self.terminal_flags
        )

    def load(self, folder_name):
        '''Loads the replay buffer from a folder'''
        self.actions = np.load(os.path.join(folder_name,'actions.npy'))
        self.timepoints = np.load(os.path.join(folder_name,'frames.npy'))
        self.rewards = np.load(os.path.join(folder_name,'rewards.npy'))
        self.terminal_flags = np.load(
            os.path.join(folder_name,'terminal_flags.npy')
        )

