"""
Iterative policy evaluation : Ref - Sutton and Barto Example 4.1
Computes the state-value function Vπ for a given policy.
"""

"""
Iterative Policy Evaluation

Computes the state-value function Vπ for a given policy.
"""

from play_8x8_world_game.environment import GridWorld8x8
from play_8x8_world_game.policy import EquiprobableRandomPolicy, Policy


def iterative_policy_evaluation(
    env: GridWorld8x8,
    policy: Policy,
    gamma: float = 1.0,
    theta: float = 1e-6,
) -> dict[int, float]:
    """
    Evaluate a policy until the value function converges.
    """

    values = {state: 0.0 for state in env.states}

    while True:
        delta = 0.0
        new_values = values.copy()

        for state in env.states:
            if env.is_terminal(state):
                continue

            new_value = 0.0

            for action in env.actions:
                action_probability = policy.probability(
                    state,
                    action,
                )

                next_state, reward, _terminated = env.step(
                    state,
                    action,
                )

                new_value += action_probability * (reward + gamma * values[next_state])

            delta = max(
                delta,
                abs(values[state] - new_value),
            )

            new_values[state] = new_value

        values = new_values

        if delta < theta:
            break

    return values


# Check above function
if __name__ == "__main__":
    env = GridWorld8x8()
    policy = EquiprobableRandomPolicy(env)

    # evaluate policy
    vpi_values = iterative_policy_evaluation(env, policy)
    print(vpi_values)
