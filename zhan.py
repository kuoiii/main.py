# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : zhan.py
# Author: JYW
# Date  : 2022/11/4

import numpy as np
import random

class FJSSP_AM_Env:
    def __init__(self, jobs, machines, processing_time, capacities, alternative_machines):
        self.jobs = jobs
        self.machines = machines
        self.processing_time = processing_time
        self.capacities = capacities
        self.alternative_machines = alternative_machines
        self.n_jobs = len(jobs)
        self.n_machines = len(machines)
        self.n_tasks = sum([len(job) for job in jobs])
        self.reset()

    def reset(self):
        self.schedule = np.zeros((self.n_jobs, self.n_machines))
        self.current_time = np.zeros((self.n_machines,))
        self.done = False
        self.curr_task = 0
        return self.schedule

    def step(self, action):
        job_idx, machine_idx = np.unravel_index(action, (self.n_jobs, self.n_machines))
        task = self.jobs[job_idx][self.curr_task]
        task_alt_machines = self.alternative_machines[task]
        if machine_idx not in task_alt_machines:
            raise ValueError("Selected machine is not an alternative for the current task")

        start_time = max(self.current_time[machine_idx], self.schedule[job_idx, machine_idx])
        completion_time = start_time + self.processing_time[task][machine_idx]
        if completion_time > self.capacities[machine_idx]:
            reward = -1
            self.done = True
        else:
            reward = 0
            self.schedule[job_idx, machine_idx] = completion_time
            self.current_time[machine_idx] = completion_time
            self.curr_task += 1
            if self.curr_task == self.n_tasks:
                self.done = True
                reward = 1

        return self.schedule, reward, self.done, {}

    def get_action_space(self):
        return np.arange(self.n_jobs * self.n_machines)

    def sample_action(self):
        return random.choice(self.get_action_space())


if __name__ == '__main__':
    jobs = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]

    machines = [0, 1, 2]

    processing_time = np.array([
        [3, 5, 2],
        [2, 1, 4],
        [5, 2, 3],
        [3, 5, 1],
        [4, 1, 2],
        [2, 4, 5],
        [5, 2, 3],
        [1, 4, 2],
        [2, 5, 3]
    ])

    capacities = np.array([10, 12, 8])

    alternative_machines = {
        0: [0, 1, 2],
        1: [0, 1],
        2: [1, 2],
        3: [0, 2],
        4: [0, 1],
        5: [0, 2],
        6: [0, 1, 2],
        7: [1, 2],
        8: [0, 1, 2]
    }

    A = FJSSP_AM_Env(jobs, machines, processing_time, capacities, alternative_machines)
    for i in range(3):
        actionspace = A.get_action_space()
        action = A.sample_action()
        print(action)
        A.step(action)
        print(A.schedule)