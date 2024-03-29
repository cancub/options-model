import json
import numpy as np
import os

import tensorflow as tf

class Agent(object):
    '''Implements a standard DDDQN agent'''

    def __init__(
        self,
        dqn,
        target_dqn,
        replay_buffer,
        n_actions,
        input_length,
        batch_size=32,
        history_length=4,
        eps_initial=1,
        eps_final=0.1,
        eps_final_frame=0.01,
        eps_evaluation=0.0,
        eps_annealing_frames=1000000,
        replay_buffer_start_size=50000,
        max_frames=25000000,
        use_per=True
    ):
        '''
        Arguments:
            dqn:
                A DQN (returned by the DQN function) to predict moves

            target_dqn:
                A DQN (returned by the DQN function) to predict target-q
                values.  This can be initialized in the same way as the dqn
                argument.

            replay_buffer:
                A ReplayBuffer object for holding all previous experiences.

            n_actions:
                Number of possible actions for the given environment

            input_length:
                Tuple/list describing the shape of the pre-processed
                environment.

            batch_size:
                Number of samples to draw from the replay memory every updating
                session.

            history_length:
                Number of historical frames available to the agent.

            eps_initial:
                Initial epsilon value.

            eps_final:
                The "half-way" epsilon value.  The epsilon value decreases more
                slowly after this.

            eps_final_frame:
                The final epsilon value.

            eps_evaluation:
                The epsilon value used during evaluation.

            eps_annealing_frames:
                Number of frames during which epsilon will be annealed to
                eps_final, then eps_final_frame.

            replay_buffer_start_size:
                Size of replay buffer before beginning to learn (after this
                many frames, epsilon is decreased more slowly).

            max_frames:
                Number of total frames the agent will be trained for.

            use_per:
                Use PER instead of classic experience replay.

        '''

        self.n_actions = n_actions
        self.input_length = input_length
        self.history_length = history_length

        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer
        self.use_per = use_per

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on
        # frame number).
        eps_diff = eps_initial - eps_final
        self.slope = -eps_diff / eps_annealing_frames
        self.intercept = eps_initial - self.slope*replay_buffer_start_size

        eps_diff_2 = eps_final - eps_final_frame
        self.slope_2 = -eps_diff_2 / (
            max_frames - eps_annealing_frames - replay_buffer_start_size)
        self.intercept_2 = eps_final_frame - self.slope_2*max_frames

        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        '''
        Get the appropriate epsilon value from a given frame number.

        Arguments:
            frame_number:
                Global frame number (used for epsilon).

            evaluation:
                True if the modesl is evaluating, False otherwise (uses
                eps_evaluation instead of default epsilon value).

        Returns:
            The appropriate epsilon value
        '''
        init_t = self.replay_buffer_start_size
        final_t = self.replay_buffer_start_size + self.eps_annealing_frames

        if evaluation:
            return self.eps_evaluation

        if frame_number < init_t:
            return self.eps_initial

        if frame_number >= init_t and frame_number < final_t:
            return self.slope*frame_number + self.intercept

        return self.slope_2*frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        '''
        Query the DQN for an action given a state.

        Arguments:
            frame_number:
                Global frame number (used for epsilon).

            state:
                State to give an action for.

            evaluation:
                True if the model is evaluating, False otherwise (uses
                eps_evaluation instead of default epsilon value).

        Returns:
            An integer as the predicted move.
        '''

        # Calculate epsilon based on the frame number.
        eps = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action.
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.DQN.predict(
            state.reshape((-1, self.input_length, self.history_length))
        )[0]
        return q_vals.argmax()

    def update_target_network(self):
        '''Update the target Q network'''
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(
        self,
        action,
        frame,
        reward,
        terminal,
        clip_reward=True
    ):
        '''
        Wrapper function for adding an experience to the Agent's replay buffer
        '''
        self.replay_buffer.add_experience(
            action,
            frame,
            reward,
            terminal,
            clip_reward
        )

    def learn(self, batch_size, gamma, frame_number, priority_scale=1.0):
        '''
        Sample a batch and use it to improve the DQN.

        Arguments:
            batch_size:
                How many samples to draw for an update.

            gamma:
                Reward discount.

            frame_number:
                Global frame number (used for calculating importances).

            priority_scale:
                How much to weight priorities when sampling the replay buffer.
                0 = completely random,
                1 = completely based on priority.

        Returns:
            The loss between the predicted and target Q as a float
        '''

        if self.use_per:
            (
                (
                    states,
                    actions,
                    rewards,
                    new_states,
                    terminal_flags
                ),
                importance,
                indices
            ) = self.replay_buffer.get_minibatch(
                batch_size=self.batch_size,
                priority_scale=priority_scale
            )
            importance = importance ** (1-self.calc_epsilon(frame_number))
        else:
            (
                states,
                actions,
                rewards,
                new_states,
                terminal_flags
            ) = self.replay_buffer.get_minibatch(
                batch_size=self.batch_size,
                priority_scale=priority_scale
            )

        # Main DQN estimates best action in new states.
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma*double_q * (1-terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients).
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            # Using tf.one_hot causes strange errors.
            one_hot_actions = tf.keras.utils.to_categorical(
                actions,
                self.n_actions,
                dtype=np.float32
            )
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also
                # scaled. The importance scale reduces bias against situataions
                # that are sampled more frequently.
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(
            zip(model_gradients, self.DQN.trainable_variables)
        )

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        '''
        Saves the Agent and all corresponding properties into a folder.

        Arguments:
            folder_name:
                Folder in which to save the Agent.

            **kwargs:
                Agent.save will also save any keyword arguments passed. This is
                used for saving the frame_number.
        '''

        # Create the folder for saving the agent.
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN.
        self.DQN.save(os.path.join(folder_name,'dqn.h5'))
        self.target_dqn.save(os.path.join(folder_name,'target_dqn.h5'))

        # Save replay buffer.
        self.replay_buffer.save(os.path.join(folder_name,'replay-buffer'))

        # Save meta.
        with open(os.path.join(folder_name,'meta.json'), 'w+') as MF:
            # Save replay_buffer information and any other information.
            kwargs.update({
                'buff_count': self.replay_buffer.count,
                'buff_curr': self.replay_buffer.current
            })
            json.dump(kwargs, MF)

    def load(self, folder_name, load_replay_buffer=True):
        '''
        Load a previously saved Agent from a folder.

        Arguments:
            folder_name:
                Folder from which to load the Agent.

        Returns:
            All other saved attributes, e.g., frame number
        '''

        if not os.path.isdir(folder_name):
            raise ValueError('{} is not a valid directory'.format(folder_name))

        # Load DQNs
        self.DQN = tf.keras.models.load_model(
            os.path.join(folder_name,'dqn.h5')
        )
        self.target_dqn = tf.keras.models.load_model(
            os.path.join(folder_name,'target_dqn.h5')
        )
        self.optimizer = self.DQN.optimizer

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(os.path.join(folder_name,'replay-buffer'))

        # Load meta
        with open(os.path.join(folder_name,'meta.json'), 'r') as F:
            meta = json.load(F)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        # we don't want to return this information
        del meta['buff_count'], meta['buff_curr']
        return meta
