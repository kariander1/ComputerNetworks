import numpy as np
import sys


class Simulation:
    def __init__(self, lambda_read, mu_read, prob_list):
        assert prob_list[0] == 1 and prob_list[-1] == 0
        assert np.all(prob_list >= 0) and np.all(prob_list <= 1)
        assert mu_read > 0
        assert lambda_read > 0
        self.lambda_read = lambda_read
        self.mu_read = mu_read
        self.N = len(prob_list)
        self.prob_list = prob_list
        self.queue_count = 0

    def run(self, T):
        self.queue_count = 0
        messages_read_count = 0
        for step in range(0, T):
            (messages_read) = self._step()
            messages_read_count += messages_read

        return ()

    def _step(self):
        incoming_messages = min(self.N - self.queue_count, self.lambda_read)
        dropped_messages = self.lambda_read - incoming_messages

        probs = np.random.rand(incoming_messages)
        valid_messages = np.count_nonzero(self.prob_list >= probs)
        faulty_messages = incoming_messages - valid_messages
        self.queue_count += valid_messages

        messages_read = min(self.mu_read, self.queue_count)
        self.queue_count -= messages_read
        return (messages_read)


if __name__ == "__main__":
    assert len(sys.argv) >= 6
    T = int(sys.argv[1])
    lambda_r = int(sys.argv[2])
    mu = int(sys.argv[3])
    prob_list = np.array(sys.argv[4:]).astype(np.float)
    sim = Simulation(lambda_r, mu, prob_list)
    sim.run(T)