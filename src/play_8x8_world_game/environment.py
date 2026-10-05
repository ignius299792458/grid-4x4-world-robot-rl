"""
RL environment

This module defines the mathematical 4x4 gridworld:

    - state space
    - terminal states
    - action space
    - transition dynamics
    - rewards

Rendering does not belong here.
"""

from enum import Enum

from play_8x8_world_game.utils import position_to_state, state_to_position


class Action(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]


class GridWorld8x8:
    GRID_SIZE = 8
    STATES_CARDINALITY = GRID_SIZE * GRID_SIZE
    STEP_REWARD = -1.0
    TERMINAL_STATE_REWARD = 0.0

    def __init__(self, terminal_states: frozenset[int] | None = None):
        self.states = tuple(range(self.STATES_CARDINALITY))
        self.actions = tuple(Action)
        self.terminal_states = (
            terminal_states if terminal_states is not None else frozenset({0, 63})
        )

        for state in self.terminal_states:
            self.validate_state(state)

    def transition(self, state: int, action: Action) -> int:
        """Execute transition in grid-env"""
        self.validate_state(state)
        self.validate_action(action)

        if self.is_terminal(state):
            return state

        x, y = state_to_position(state, self.GRID_SIZE)
        next_x = x + action.dx
        next_y = y + action.dy
        if not self._is_valid_position(next_x, next_y):
            return state
        return position_to_state(next_x, next_y, self.GRID_SIZE)

    def step(self, state: int, action: Action) -> tuple[int, float, bool]:
        """Execute one MDP transition, return: (next_state, reward, terminated)"""
        self.validate_state(state)
        self.validate_action(action)

        if self.is_terminal(state):
            return state, self.TERMINAL_STATE_REWARD, True

        next_state = self.transition(state, action)
        reward = self.STEP_REWARD
        terminated = self.is_terminal(next_state)

        return next_state, float(reward), terminated

    def is_terminal(self, state: int) -> bool:
        self.validate_state(state)
        return state in self.terminal_states

    def _is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE

    def validate_state(self, state: int) -> None:
        if state not in self.states:
            raise ValueError(
                f"State {state} is outside "
                f"the valid value range 0-{self.STATES_CARDINALITY - 1}."
            )

    def validate_action(self, action: Action) -> None:
        if not isinstance(action, Action):
            raise ValueError(f"{action!r} is not a valid Action.")
