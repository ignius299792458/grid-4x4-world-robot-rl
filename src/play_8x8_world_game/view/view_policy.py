import matplotlib.pyplot as plt

ACTION_SYMBOLS = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→",
}


def plot_improved_policy(
    values: dict[int, float], best_actions: dict[int, tuple], grid_size: int = 4
):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw empty grid
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for state in range(16):
        x = state % grid_size
        y = state // grid_size

        value_text = f"{values[state]:.1f}"

        if state in (0, grid_size * grid_size - 1):
            action_text = "T"
        else:
            symbols = [ACTION_SYMBOLS[action.name] for action in best_actions[state]]
            action_text = "/".join(symbols)

        # Write state id
        ax.text(
            x + 0.08,
            y + 0.20,
            f"s={state}",
            fontsize=9,
        )

        # Write value
        ax.text(
            x + 0.5,
            y + 0.45,
            value_text,
            ha="center",
            va="center",
            fontsize=11,
        )

        # Write best action(s)
        ax.text(
            x + 0.5,
            y + 0.78,
            action_text,
            ha="center",
            va="center",
            fontsize=14,
        )

    plt.title("Greedy Policy and State Values")
    plt.show()


def plot_iterated_policy(
    action_map: dict[int, tuple],
    values: dict[int, float],
    grid_size: int = 4,
    title: str = "Policy Iteration Result",
) -> None:
    action_symbols = {
        "UP": "↑",
        "DOWN": "↓",
        "LEFT": "←",
        "RIGHT": "→",
    }

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True, linewidth=1.5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for state in range(grid_size * grid_size):
        x = state % grid_size
        y = state // grid_size

        value = values[state]
        best_actions = action_map[state]

        if len(best_actions) == 0:
            action_text = "T"
        else:
            action_text = " ".join(
                action_symbols[action.name] for action in best_actions
            )

        # state id
        ax.text(
            x + 0.08,
            y + 0.18,
            f"s={state}",
            fontsize=10,
        )

        # state value
        ax.text(
            x + 0.5,
            y + 0.48,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=12,
        )

        # best action(s)
        ax.text(
            x + 0.5,
            y + 0.78,
            action_text,
            ha="center",
            va="center",
            fontsize=16,
        )

    ax.set_title(title)
    plt.show()
