import math
import random
import numpy as np
import sys


class Event:
    def __init__(self, time):
        # C'tor for updating time
        self.time = time

    def dequeue(self, sim):
        # Update sims number of time steps with queue size
        sim.time_counts[sim.queue_count] += self.time - sim.last_event_time

    def __repr__(self):
        return "Event Time {}".format(self.time)


class EventService(Event):
    # Class for representing an end of service to a message
    def dequeue(self, sim):
        # Call event's dequeue
        super().dequeue(sim)
        # Update sim's internals
        sim.queue_count -= 1
        sim.n_read_messages += 1

    def __repr__(self):
        return "Save Message " + super().__repr__()


class EventIncomingMessage(Event):
    # Class representing an incoming message event
    def dequeue(self, sim):
        # Call event class dequeue
        super().dequeue(sim)

        # Generate a probability in U[0,1)
        prob = min(random.random(), 1 - sys.float_info.epsilon)
        if prob < prob_list[sim.queue_count]:
            # Message gets inserted to queue
            sim.queue_count += 1
            start_service_time = max(self.time, sim.latest_service_time)
            service_event = generate_message(sim.mu_read, EventService)

            # Calc service time
            sim.total_service_time += service_event.time

            # Add the process time to the start service to get
            # When the event of service is finished
            service_event.time += start_service_time
            
            # Update sim's internals
            sim.latest_service_time = service_event.time
            
            # Insert service event sorted into queue
            sim.insert_sorted(service_event)

            # Calc wait time
            sim.total_wait_time += (start_service_time - self.time)

            # Calc receive rate
            sim.total_insert_time += self.time - sim.last_insert_time
            sim.last_insert_time = self.time
        else:
            # Message is faulty
            sim.n_fault_messages += 1

    def __repr__(self):
        return "Insert " + super().__repr__()


def generate_message(lambda_read, EventType):
    # Global method for creating an event with time difference
    # Use approximation of `exp(lambda)=-ln(U)/lambda`
    time = -math.log(random.random()) / lambda_read
    return EventType(time)


class Simulation:

    def insert_sorted(self, event):
        index = 0
        while index < len(self.event_queue) and event.time >= self.event_queue[index].time: index += 1
        self.event_queue.insert(max(index, 0), event)

    def create_insert_events(self, T):
        return [EventIncomingMessage(i / self.lambda_read) for i in range(1, int(T // (1 / self.lambda_read)))]

    def __init__(self, lambda_read, mu_read, prob_list):
        # Assert inputs to sim
        assert prob_list[0] == 1 and prob_list[-1] == 0
        assert np.all(prob_list >= 0) and np.all(prob_list <= 1)
        assert mu_read >= 0
        assert lambda_read >= 0

        # Create self attribute for simulator
        self.lambda_read = lambda_read
        self.mu_read = mu_read
        self.N = len(prob_list)
        self.prob_list = prob_list
        self.reset_sim()

    def reset_sim(self):
        self.queue_count = 0
        self.event_queue = []
        self.time_counts = np.zeros(len(self.prob_list))
        self.latest_service_time = 0
        self.last_event_time = 0
        self.n_read_messages = 0
        self.n_fault_messages = 0
        self.total_service_time = 0
        self.total_wait_time = 0
        self.total_insert_time = 0
        self.last_insert_time = 0

    def run(self, T):
        self.reset_sim()

        # Create a first insert event to the queue
        self.event_queue = [generate_message(self.lambda_read, EventIncomingMessage)]

        # init t_prime
        t_prime = 0

        # Iterate as long as there are events
        while self.event_queue:
            # Take event from top of the queue
            event = self.event_queue.pop(0)
            # perform the event dequeue method
            event.dequeue(self)
            self.last_event_time = event.time

            # Create next message event
            next_message = generate_message(self.lambda_read, EventIncomingMessage)
            next_message.time += event.time

            if next_message.time <= T and isinstance(event, EventIncomingMessage):
                # Insert to queue only if it is before T
                self.insert_sorted(next_message)
            elif isinstance(event, EventService):
                # If it is a service event, update T'
                t_prime = event.time

        # Calculate requested results
        z_i = [self.time_counts[i] / t_prime for i in range(len(self.time_counts))]
        T_w = self.total_wait_time / self.n_read_messages
        T_s = self.total_service_time / self.n_read_messages
        lambda_a = self.n_read_messages / self.total_insert_time
        
        # Encapsulate results
        return tuple(
            [self.n_read_messages, self.n_fault_messages, t_prime] + self.time_counts.tolist() + z_i + [T_w, T_s,
                                                                                                        lambda_a])


if __name__ == "__main__":
    # Expect at least 6 args
    assert len(sys.argv) >= 6

    # Parse sim time
    T = int(sys.argv[1])

    # Parse sim read and serve rates
    lambda_r = int(sys.argv[2])
    mu = int(sys.argv[3])
    # Parse probabilities as numpy array
    prob_list = np.array(list(map(float, sys.argv[4:])))

    # Create sim instance, run and print results
    sim = Simulation(lambda_r, mu, prob_list)
    results = sim.run(T)
    print(' '.join(str(x) for x in results))
