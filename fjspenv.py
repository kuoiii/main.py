# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : fjspenv.py
# Author: JYW
# Date  : 2022/10/16


import numpy as np

class JobShopEnv:
    def __init__(self, num_jobs, num_machines, processing_time):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_time = processing_time
        self.state = np.zeros((num_jobs, num_machines))

    def reset(self):
        self.state = np.zeros((self.num_jobs, self.num_machines))
        return self.state

    def step(self, action):
        job, machine = action
        self.state[job, machine] = self.processing_time[job, machine]
        done = np.all(self.state > 0)
        reward = -1
        if done:
            reward = 0
        return self.state, reward, done

/////
import numpy as np
import random

class JobShopEnv:
    def __init__(self, num_jobs, num_machines, processing_time):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_time = processing_time
        self.state = np.zeros((num_jobs, num_machines))
        self.completion_time = np.zeros((num_jobs,))

    def reset(self):
        self.state = np.zeros((self.num_jobs, self.num_machines))
        self.completion_time = np.zeros((self.num_jobs,))
        return self.state

    def step(self, action):
        job, machine = action
        if self.state[job, machine] == 0:
            processing_time = self.processing_time[job, machine]
            completion_time = np.max([self.completion_time[m] for m in range(self.num_machines) if self.state[job, m] > 0]) + processing_time
            self.state[job, machine] = processing_time
            self.completion_time[job] = completion_time
            done = np.all(self.state[job, :] > 0)
            reward = -1
            if done:
                reward = -completion_time
            return self.state, reward, done, {}
        else:
            return self.state, 0, False, {}

    def sample_action(self):
        job = random.choice(range(self.num_jobs))
        machine = random.choice([m for m in range(self.num_machines) if self.state[job, m] == 0])
        return job, machine
    ///
    import numpy as np
    import random

    class FJSSPEnv:
        def __init__(self, num_jobs, num_machines, job_durations, job_sequences, machine_capacities):
            self.num_jobs = num_jobs
            self.num_machines = num_machines
            self.job_durations = job_durations
            self.job_sequences = job_sequences
            self.machine_capacities = machine_capacities
            self.state = np.zeros((num_jobs, num_machines), dtype=int)
            self.done = False
            self.current_time = 0

        def reset(self):
            self.state = np.zeros((self.num_jobs, self.num_machines), dtype=int)
            self.done = False
            self.current_time = 0
            return self.state

        def step(self, action):
            # Get the job and machine indices from the action
            job_index, machine_index = action
            job_duration = self.job_durations[job_index][machine_index]

            # Check if the machine is available
            machine_capacity = self.machine_capacities[machine_index]
            if np.sum(self.state[:, machine_index]) + job_duration > machine_capacity:
                reward = -100
                self.done = True
            else:
                # Assign the job to the machine and update the state
                self.state[job_index][machine_index] = job_duration
                self.current_time += job_duration
                reward = -self.current_time

                # Check if all jobs are completed
                if np.sum(self.state) == np.sum(self.job_durations):
                    self.done = True

            return self.state, reward, self.done, {}

        def sample_action(self):
            # Sample a random job and machine from the action space
            job_index = random.randint(0, self.num_jobs - 1)
            machine_index = self.job_sequences[job_index][self.state[job_index].nonzero()[0].shape[0]]
            return job_index, machine_index