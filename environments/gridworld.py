
class GridWorld:
    def __init__(self, size=4, start=(0, 0), goal=(3, 3), step_reward=-1):
        self.size = size
        self.start = start
        self.goal = goal
        self.step_reward = step_reward

        self.agent_position = self.start
        self.actions = ["up", "down", "left", "right"]
        self.done = False

    def get_next_position(self, state, action):
        row, col = state

        if action == "up":
            row -= 1
        elif action == "down":
            row += 1
        elif action == "left":
            col -= 1
        elif action == "right":
            col += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        row = max(0, min(row, self.size - 1))
        col = max(0, min(col, self.size - 1))

        return (row, col)

    def reset(self):
        self.agent_position = self.start
        self.done = False
        return self.agent_position

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )

        next_position = self.get_next_position(
            self.agent_position, action
        )

        self.agent_position = next_position

        reward = self.step_reward
        self.done = self.agent_position == self.goal

        return self.agent_position, reward, self.done
