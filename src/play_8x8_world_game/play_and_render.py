"""
Play and Render — Sutton & Barto Example 4.1

This file connects:

    RL Environment
        ↓
    Current Policy
        ↓
    Policy Evaluation
        ↓
    Policy Improvement
        ↓
    GridWorld Renderer

The renderer does NOT perform learning.

The RL environment and dynamic-programming algorithms compute
the policy. The renderer only demonstrates the behavior of the
current policy.
"""

import random
import time
from pathlib import Path

from grid_nxn_world import GridWorld
from play_8x8_world_game.environment import Action, GridWorld8x8
from play_8x8_world_game.policy import EquiprobableRandomPolicy, Policy
from play_8x8_world_game.policy_evaluation import iterative_policy_evaluation
from play_8x8_world_game.policy_improvement import improve_greedy_policy
from play_8x8_world_game.policy_iteration import check_policies_equal
from play_8x8_world_game.utils import state_to_position

# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

TERMINAL_STATES = frozenset(
    {
        0,
        15,
        30,
        45,
        63,
    }
)


# ---------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------

MAX_EPISODE_STEPS = 100

MOVE_DISPLAY_SECONDS = 0.45
START_DISPLAY_SECONDS = 0.5
END_DISPLAY_SECONDS = 1.0

GAMMA = 1.0
THETA = 1e-6


# ---------------------------------------------------------
# Assets
# ---------------------------------------------------------

GOAL_SOUND = Path(__file__).parent / "assets" / "goal.wav"


# ---------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------


def get_random_start_state(
    env: GridWorld8x8,
) -> int:
    """
    Select a random nonterminal state.

    Episodes must never begin in a terminal state.
    """

    nonterminal_states = [state for state in env.states if not env.is_terminal(state)]
    return random.choice(nonterminal_states)


def configure_terminal_cells(
    env: GridWorld8x8,
    world: GridWorld,
) -> None:
    """
    Convert RL terminal states into renderer coordinates
    and visually mark those grid cells.
    """

    positions = tuple(
        state_to_position(
            state,
            env.GRID_SIZE,
        )
        for state in env.terminal_states
    )

    world.grid_set_terminal_positions(positions)


def configure_goal_sound(
    world: GridWorld,
) -> None:
    """
    Load goal sound if the asset exists.

    The RL environment does not know anything about sound.
    """

    if not GOAL_SOUND.exists():
        print(f"Goal sound not found: {GOAL_SOUND}")
        return

    world.render_load_goal_sound(str(GOAL_SOUND))


# ---------------------------------------------------------
# Policy action sampling
# ---------------------------------------------------------


def choose_action(
    env: GridWorld8x8,
    policy: Policy,
    state: int,
) -> Action:
    """
    Sample one action according to π(a|s).

    Works with:
        - EquiprobableRandomPolicy
        - improved greedy policies
    """

    actions: list[Action] = []
    probabilities: list[float] = []

    for action in env.actions:
        probability = policy.probability(
            state,
            action,
        )

        if probability > 0.0:
            actions.append(action)
            probabilities.append(probability)

    if not actions:
        raise RuntimeError(
            f"Policy has no available action " f"for nonterminal state {state}."
        )

    return random.choices(
        actions,
        weights=probabilities,
        k=1,
    )[0]


# ---------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------


def render_for(
    world: GridWorld,
    seconds: float,
) -> bool:
    """
    Keep rendering for a fixed amount of time.

    Returns False when the user closes the window.
    """

    end_time = time.perf_counter() + seconds

    while time.perf_counter() < end_time:
        if not world.render():
            return False

    return True


def show_state(
    world: GridWorld,
    state: int,
    grid_size: int,
    *,
    animate: bool = True,
) -> None:
    """
    Move the rendered robot to an RL state.
    """

    x, y = state_to_position(
        state,
        grid_size,
    )

    world.robot_set_position(
        x,
        y,
        animate=animate,
    )


# ---------------------------------------------------------
# Episode
# ---------------------------------------------------------


def play_episode(
    env: GridWorld8x8,
    policy: Policy,
    world: GridWorld,
    start_state: int,
    iteration: int,
) -> bool:

    state = start_state

    # Store RL-state path for console inspection.
    path: list[int] = [state]

    print()
    print(f"Playing policy from iteration {iteration} " f"starting at state {state}")

    # -----------------------------------------------------
    # Start a fresh trace
    # -----------------------------------------------------

    world.robot_trace_clear()

    show_state(
        world,
        state,
        env.GRID_SIZE,
        animate=False,
    )

    world.robot_trace_add()

    if not render_for(
        world,
        START_DISPLAY_SECONDS,
    ):
        return False

    # -----------------------------------------------------
    # Episode transitions
    # -----------------------------------------------------

    # for step_number in range(
    #     1,
    #     MAX_EPISODE_STEPS + 1,
    # ):
    step_number = 1
    while True:

        if env.is_terminal(state):
            break

        action = choose_action(
            env,
            policy,
            state,
        )

        next_state, reward, terminated = env.step(
            state,
            action,
        )

        path.append(next_state)

        print(
            f"step={step_number:02d} | "
            f"state={state:2d} | "
            f"action={action.name:<5} | "
            f"next={next_state:2d} | "
            f"reward={reward:4.1f}"
        )

        # -------------------------------------------------
        # Move robot
        # -------------------------------------------------

        show_state(
            world,
            next_state,
            env.GRID_SIZE,
            animate=True,
        )

        # Add the new cell to this episode's trace.
        world.robot_trace_add()

        if not render_for(
            world,
            MOVE_DISPLAY_SECONDS,
        ):
            return False

        state = next_state

        # -------------------------------------------------
        # Goal reached
        # -------------------------------------------------

        if terminated:
            print("-" * 60)
            print(f"GOAL REACHED: terminal state {state}")
            print(f"Episode completed in " f"{step_number} steps.")
            print(
                "Path:",
                " -> ".join(str(s) for s in path),
            )
            print("-" * 60)
            world.render_play_goal_sound()

            world.robot_trace_clear()
            return render_for(
                world,
                END_DISPLAY_SECONDS,
            )

    # -----------------------------------------------------
    # Episode exceeded demonstration limit
    # -----------------------------------------------------

    print("-" * 60)
    print(f"Episode stopped after " f"{MAX_EPISODE_STEPS} steps.")
    print(
        "Path:",
        " -> ".join(str(s) for s in path),
    )
    print("-" * 60)

    return render_for(
        world,
        END_DISPLAY_SECONDS,
    )


# ---------------------------------------------------------
# Console inspection
# ---------------------------------------------------------


def print_values(
    env: GridWorld8x8,
    values: dict[int, float],
) -> None:
    """
    Print Vπ using the same spatial layout as the grid.
    """

    print("\nState values Vπ:")

    for row in range(env.GRID_SIZE):

        row_values = []

        for column in range(env.GRID_SIZE):
            state = row * env.GRID_SIZE + column

            row_values.append(f"{values[state]:7.2f}")

        print(" ".join(row_values))


# ---------------------------------------------------------
# Complete visual policy iteration
# ---------------------------------------------------------


def play_and_render() -> None:
    """
    Run and visually demonstrate complete policy iteration.

    Flow:

        π₀
        ↓
        Policy Evaluation
        ↓
        Demonstrate π₀
        ↓
        Policy Improvement
        ↓
        π₁
        ↓
        Policy Evaluation
        ↓
        Demonstrate π₁
        ↓
        ...
        ↓
        Stable Policy
    """

    # -----------------------------------------------------
    # RL environment
    # -----------------------------------------------------

    env = GridWorld8x8(
        terminal_states=TERMINAL_STATES,
    )

    initial_state = get_random_start_state(env)

    # -----------------------------------------------------
    # Rendering world
    # -----------------------------------------------------

    world = GridWorld(
        grid_size=env.GRID_SIZE,
        robot_start_position=state_to_position(
            initial_state,
            env.GRID_SIZE,
        ),
        render_title=("Sutton & Barto — Policy Iteration"),
    )

    # Visually mark RL terminal states.
    configure_terminal_cells(
        env,
        world,
    )

    # Optional goal sound.
    configure_goal_sound(world)

    # -----------------------------------------------------
    # Initial policy π₀
    # -----------------------------------------------------

    policy: Policy = EquiprobableRandomPolicy(env)

    iteration = 0

    try:

        while True:

            print()
            print("=" * 60)
            print(f"POLICY ITERATION {iteration}")
            print("=" * 60)

            # -------------------------------------------------
            # 1. POLICY EVALUATION
            # -------------------------------------------------

            print("\nEvaluating current policy...")

            values = iterative_policy_evaluation(
                env=env,
                policy=policy,
                gamma=GAMMA,
                theta=THETA,
            )

            print_values(
                env,
                values,
            )

            # -------------------------------------------------
            # 2. DEMONSTRATE CURRENT POLICY
            # -------------------------------------------------

            print("\nDemonstrating current policy...")

            if not play_episode(
                env=env,
                policy=policy,
                world=world,
                start_state=get_random_start_state(env),
                iteration=iteration,
            ):
                return

            # -------------------------------------------------
            # 3. POLICY IMPROVEMENT
            # -------------------------------------------------

            print("\nImproving policy...")

            improved_policy = improve_greedy_policy(
                env=env,
                vpi_values=values,
                gamma=GAMMA,
            )

            # -------------------------------------------------
            # 4. POLICY STABILITY
            # -------------------------------------------------

            stable = check_policies_equal(
                env,
                policy,
                improved_policy,
            )

            if stable:

                print()
                print("=" * 60)
                print("POLICY IS STABLE")
                print("=" * 60)

                print("\nFinal policy demonstration...")

                if not play_episode(
                    env=env,
                    policy=improved_policy,
                    world=world,
                    start_state=get_random_start_state(env),
                    iteration=iteration,
                ):
                    return

                print("\nPolicy iteration finished.")

                break

            print(
                "\nPolicy changed." "\nImproved policy becomes " "the current policy."
            )

            # -------------------------------------------------
            # πₖ ← πₖ₊₁
            # -------------------------------------------------

            policy = improved_policy

            iteration += 1

    finally:
        world.render_close()


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------


if __name__ == "__main__":
    play_and_render()
