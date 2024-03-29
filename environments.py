# Python
import numpy as np
import random

# Local
import config
from state_managers import StrategyStateManager

# Tensorflow
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class OptionsTradingEnv(py_environment.PyEnvironment):

    def __init__(
        self,
        ticker,
        expiries,
        paddlin=0,
        bonus=0,
        history_length=4,
        no_op_steps=10,
        max_margin=config.MARGIN,
        vertical=True,
        butterfly=True,
        gen_processes=8,
        queue_size=32,
    ):
        # This is a ready-made state manager which has already loaded
        # up a strategy generator as well as the first state.
        self.state_manager = StrategyStateManager(
            ticker,
            expiries,
            max_margin=max_margin,
            vertical=vertical,
            butterfly=butterfly,
            gen_processes=gen_processes,
            queue_size=queue_size
        )

        self.input_length = self.state_manager.state.shape[0]
        self.no_op_steps = no_op_steps

        self.history_length = history_length

        self.strat_timepoint = None

        self.episode_ended = False

        self.bonus = bonus
        self.paddlin = paddlin

        # We can have 2 actions:
        #   0: do nothing
        #   1: toggle the strategy (buy if not holding, sell if holding)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')

        # Remember that observations will include several timepoints.
        self._observation_spec = array_spec.ArraySpec(
            shape=(self.input_length*self.history_length,),
            dtype=np.float64,
            name='observation'
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self, evaluation=False):
        '''
        Resets the environment

        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a
                        random number of no-op steps if True.

        Returns:
            A "restart" TimeStep with the current state.
        '''
        self.episode_ended = False
        self.state_manager.reset()

        # If evaluating, take a random number of no-op steps. This adds an
        # element of randomness, so that the each evaluation is slightly
        # different.
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.strat_timepoint = self.state_manager.step()

        # For the initial state, we tile the first timepoint four times.
        self._state = np.tile(self.strat_timepoint, self.history_length)

        return ts.restart(self._state)

    def _step(self, action):

        if self.episode_ended:
            # The last action ended the episode. Ignore the current action and
            # start a new episode.
            return self.reset()

        # We're performing an action. Check to see if this is the last
        # possible moment to do anything.
        if (self.state_manager.over
                or (self.state_manager.holding and action == 1)):
            self.episode_ended = True
            # The episode is over, so we need to compare how we did vs the best
            # possible profit. This is 0 if it's a bad trade and we decide to
            # sit out.
            max_profit = max(self.state_manager.max_profit, 0)

        # Work from the base of a zero reward.
        reward = 0

        # Make sure episodes don't go on forever.
        if action == 1:
            # Making a transition in what we're holding.
            if not self.state_manager.holding:
                # Buying the strategy.
                # Is there even any time left to buy?
                if self.state_manager.over:
                    # The agent just attempted to buy the strategy when there
                    # was no time left to do so. There's a difference between
                    # being bold and being dumb. That's a paddlin'.
                    reward += self.paddlin
                else:
                    # Let the state manager know to copy the current strategy
                    # into the stored strategy section of the state array.
                    try:
                        self.state_manager.buy()
                    except (ValueError, AttributeError):
                        # This trade wasn't actually available. Shame on the
                        # agent for trying to buy it.
                        reward += self.paddlin
                    else:
                        # Reward boldness.
                        reward += self.bonus
            else:
                # Selling the strategy.
                try:
                    # How close was the agent to the maximum profit (including
                    # if they had sat out a strategy that had no way to win).
                    profit = self.state_manager.sell()
                    reward += profit - max_profit
                except AttributeError:
                    # There wasn't a closing trade available.
                    reward += self.paddlin
                else:
                    # Then sweeten the deal if the agent sold prior to the
                    # expiry.
                    reward += self.bonus
        elif action == 0:
            # Sticking with what we're currently holding (if anything).
            if self.state_manager.over:
                # Don't hold the strategy to the bitter end.
                reward += self.paddlin

                # In the case of the agent holding a strategy at we'll give
                # them the benefit of the doubt and sell for them (factoring in
                # the penalty).
                if self.state_manager.holding:
                    reward += self.state_manager.sell(allow_nan=True)

                # Be sure to compare this to what they would have made in
                # the ideal scenario (including if they had sat out a strategy
                # that had no way to win).
                reward -= max_profit
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self.episode_ended:
            return ts.termination(self._state, reward)
        else:
            # Move on to the next timepoint.
            self.strat_timepoint = self.state_manager.step()
            self._state = np.append(
                self._state[:-self.input_length],
                self.strat_timepoint
            )
            return ts.transition(self._state, reward=reward, discount=1.0)
