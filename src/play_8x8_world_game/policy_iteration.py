"""Policy Iteration
Repeatedly:
    1. evaluate current policy
    2. improve current policy
    3. stop when policy becomes stable
"""

from math import isclose

from play_8x8_world_game.environment import GridWorld8x8
from play_8x8_world_game.policy import EquiprobableRandomPolicy, GreedyPolicy, Policy
from play_8x8_world_game.policy_evaluation import iterative_policy_evaluation
from play_8x8_world_game.policy_improvement import improve_greedy_policy


def policy_iteration(
    env: GridWorld8x8,
    gamma: float = 1.0,
    theta: float = 1e-6,
) -> tuple[GreedyPolicy, dict[int, float], int]:

    policy: Policy = EquiprobableRandomPolicy(env=env)
    iteration = 0

    while True:
        iteration += 1

        """ 1. Policy Evaluation """
        vpi_values = iterative_policy_evaluation(env, policy, gamma, theta)

        """ 2. Policy Improvement 
        (first π0 - equiprobable then it turned into greedy policy from π1 -> greedy) """
        improved_policy = improve_greedy_policy(env, vpi_values, gamma)

        """ 3. check whether policy changed """
        if check_policies_equal(env, policy, improved_policy):
            return improved_policy, vpi_values, iteration

        policy = improved_policy


def check_policies_equal(env: GridWorld8x8, policy_a: Policy, policy_b: Policy) -> bool:

    for state in env.states:
        if env.is_terminal(state):
            continue

        for action in env.actions:
            probability_a = policy_a.probability(
                state,
                action,
            )

            probability_b = policy_b.probability(
                state,
                action,
            )

            if not isclose(
                probability_a,
                probability_b,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return False
    return True


# Check and test policy_iteration function
if __name__ == "__main__":
    iterated_improved_policy, vpi_values, iterations = policy_iteration(
        env=GridWorld8x8()
    )
    print(f"{iterated_improved_policy=},\n{vpi_values=}\n{iterations=}")
