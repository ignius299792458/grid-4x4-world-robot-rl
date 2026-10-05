def state_to_position(state: int, grid_size: int) -> tuple[int, int]:
    x = state % grid_size
    y = state // grid_size
    return x, y


def position_to_state(x: int, y: int, grid_size: int) -> int:
    return x + grid_size * y
