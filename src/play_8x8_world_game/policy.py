"""
Policy

1. Equiprobable Random Policy:
   each action has equal probability.
"""

from abc import ABC, abstractmethod

from play_8x8_world_game.environment import Action, GridWorld8x8


class Policy(ABC):
    """Abstract policy contract."""

    def __init__(self, env: GridWorld8x8):
        self._env = env

    @abstractmethod
    def probability(
        self,
        state: int,
        action: Action,
    ) -> float:
        """
        Return π(a | s):
        probability of taking action a in state s.
        """
        raise NotImplementedError

    def __repr__(self):
        return "Policy : define your required policy"


class EquiprobableRandomPolicy(Policy):
    def __init__(self, env: GridWorld8x8):
        super().__init__(env)

        self._action_probability = 1.0 / len(self._env.actions)

    def probability(
        self,
        state: int,
        action: Action,
    ) -> float:
        self._env.validate_state(state)
        self._env.validate_action(action)

        if self._env.is_terminal(state):
            return 0.0

        return self._action_probability

    def __repr__(self):
        return (
            f"EquiprobableRandomPolicy : action_probability={self._action_probability}"
        )


class GreedyPolicy(Policy):
    def __init__(self, env, action_map: dict[int, tuple[Action, ...]]):
        super().__init__(env)
        self._action_map = action_map  # give {state: possible better action set}

    @property
    def action_map(self):
        return self._action_map

    def probability(self, state: int, action: Action) -> float:
        self._env.validate_state(state)
        self._env.validate_action(action)

        if self._env.is_terminal(state):
            return 0.0

        best_actions = self._action_map[state]
        if action not in best_actions:
            return 0.0
        return 1.0 / len(best_actions)

    def actions(self, state: int) -> tuple[Action, ...]:
        self._env.validate_state(state)

        if self._env.is_terminal(state):
            return ()
        return self._action_map[state]

    def __repr__(self):
        return f"GreedyPolicy : action_map={self._action_map}"
