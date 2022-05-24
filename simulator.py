import math
import random
import numpy as np
import sys


class Event:
    def __init__(self, time):
        self.time = time

    def dequeue(self, sim):
        sim.time_counts[sim.queue_count] += self.time - sim.last_event_time

    def __repr__(self):
        return "Event Time {}".format(self.time)


class EventSaveMessage(Event):
    def dequeue(self, sim):
        super().dequeue(sim)
        sim.queue_count -= 1
        sim.n_read_messages += 1

    def __repr__(self):
        return "Save Message " + super().__repr__()


class EventInsert(Event):
    def dequeue(self, sim):
        super().dequeue(sim)
        prob = min(random.random(), 1 - sys.float_info.epsilon)  # Generate U[0,1)
        if prob < prob_list[sim.queue_count]:
            # Message gets inserted
            sim.queue_count += 1
            finish_process_time = max(self.time, sim.last_process_time)
            process_event = generate_message(sim.mu_read, EventSaveMessage)

            # Calc service time
            sim.total_service_time += process_event.time

            process_event.time += finish_process_time
            sim.last_process_time = process_event.time
            sim.insert_sorted(process_event)

            # Calc wait time
            sim.total_wait_time += (finish_process_time - self.time)

            # Calc receive rate
            sim.total_insert_time += self.time - sim.last_insert_time
            sim.last_insert_time = self.time
        else:
            sim.n_fault_messages += 1

    def __repr__(self):
        return "Insert " + super().__repr__()


def generate_message(lambda_read, EventType):
    time = -math.log(random.random()) / lambda_read  # exp(lambda)=-ln(U)/lambda
    return EventType(time)


class Simulation:

    def insert_sorted(self, event):
        index = 0
        while index < len(self.event_queue) and event.time >= self.event_queue[index].time: index += 1
        self.event_queue.insert(max(index, 0), event)

    def create_insert_events(self, T):
        return [EventInsert(i / self.lambda_read) for i in range(1, int(T // (1 / self.lambda_read)))]

    def __init__(self, lambda_read, mu_read, prob_list):
        assert prob_list[0] == 1 and prob_list[-1] == 0
        assert np.all(prob_list >= 0) and np.all(prob_list <= 1)
        assert mu_read > 0
        assert lambda_read > 0

        self.lambda_read = lambda_read
        self.mu_read = mu_read
        self.N = len(prob_list)
        self.prob_list = prob_list
        self.reset_sim()

    def reset_sim(self):
        self.queue_count = 0
        self.event_queue = []
        self.time_counts = np.zeros(len(self.prob_list))
        self.last_process_time = 0
        self.last_event_time = 0
        self.n_read_messages = 0
        self.n_fault_messages = 0
        self.total_service_time = 0
        self.total_wait_time = 0
        self.total_insert_time = 0
        self.last_insert_time = 0

    def run(self, T):
        self.reset_sim()
        self.event_queue = [generate_message(self.lambda_read, EventInsert)]
        t_prime = 0

        while self.event_queue:
            event = self.event_queue.pop(0)
            event.dequeue(self)
            self.last_event_time = event.time

            # Create next message event
            next_message = generate_message(self.lambda_read, EventInsert)
            next_message.time += event.time
            if next_message.time <= T and isinstance(event, EventInsert):
                # Insert to queue only if it is before T
                self.insert_sorted(next_message)
            elif isinstance(event, EventSaveMessage):
                t_prime = event.time
        z_i = [self.time_counts[i] / t_prime for i in range(len(self.time_counts))]
        T_w = self.total_wait_time / self.n_read_messages
        T_s = self.total_service_time / self.n_read_messages
        lambda_a = self.n_read_messages / self.total_insert_time
        return tuple(
            [self.n_read_messages, self.n_fault_messages, t_prime] + self.time_counts.tolist() + z_i + [T_w, T_s,
                                                                                                        lambda_a])


if __name__ == "__main__":
    assert len(sys.argv) >= 6
    T = int(sys.argv[1])
    lambda_r = int(sys.argv[2])
    mu = int(sys.argv[3])
    prob_list = np.array(list(map(float, sys.argv[4:])))
    sim = Simulation(lambda_r, mu, prob_list)
    print(sim.run(T))
