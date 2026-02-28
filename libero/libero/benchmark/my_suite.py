
class my_suite:
    """
    Minimal suite stub. We'll add real tasks and init states next.
    """
    def __init__(self, data_root=None, n_trials=50):
        self.n_tasks = 0
        self.tasks = []
        self.data_root = data_root
        self.n_trials = n_trials

    def get_task(self, idx: int):
        raise IndexError("No tasks yet in my_suite.")

    def get_task_init_states(self, idx: int):
        raise IndexError("No init states yet in my_suite.")
