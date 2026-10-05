"""Policy Improvements"""

from math import isclose

from play_8x8_world_game.environment import Action, GridWorld8x8
from play_8x8_world_game.policy import GreedyPolicy


def action_value(
    env: GridWorld8x8,
    vpi_values: dict[int, float],
    state: int,
    action: Action,
    gamma: float = 0.1,
) -> float:
    next_state, reward, _terminated = env.step(state, action)
    return reward + gamma * vpi_values[next_state]


def gready_actions(
    env: GridWorld8x8,
    vpi_values: dict[int, float],
    state: int,
    gamma: float = 1.0,
) -> tuple[Action, ...]:
    """Return all actions having the highest action value"""

    env.validate_state(state)

    if env.is_terminal(state):
        return ()

    action_values = {
        action: action_value(
            env=env, vpi_values=vpi_values, state=state, action=action, gamma=gamma
        )
        for action in Action
    }

    best_value = max(action_values.values())

    return tuple(
        action
        for action, value in action_values.items()
        if isclose(
            value,
            best_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )


def improve_greedy_policy(
    env: GridWorld8x8, vpi_values: dict[int, float], gamma: float = 1.0
) -> GreedyPolicy:
    action_map = {
        state: gready_actions(env, vpi_values, state, gamma) for state in env.states
    }
    return GreedyPolicy(env, action_map)
