class Env:
    def __init__(self):
        self.current = 0
        self.reward = [1, 5]

    def _switch_state(self):
        self.current = 1 if self.current == 0 else 0

    def move(self, action):
        if action == 'L' and self.current == 1:
            self._switch_state()
        elif action == 'R' and self.current == 0:
            self._switch_state()

        if np.random.random() > 0.9:
            self._switch_state()

