# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : fjsp-deter-main.py
# Author: JYW
# Date  : 2023/2/15

import random
import numpy as np


class FJSP:
    def __init__(self, n_jobs, n_machines, processing_times, h1,h2 ):
        self.n_jobs = n_jobs                  #工件数量
        self.n_machines = n_machines          #机器数量
        self.processing_times = processing_times   #  加工时间 processing_times[0][1][0]代表第0个工件的第1道工序在第0台机器上进行加工所需时间
        self.time = 0
        self.job_status = [1 for _ in range(self.n_jobs)]               #工件的工序状态
        self.machine_status = [0 for _ in range(self.n_machines)]           #机器的加工时间状态
        self.done = False
        self.prev_job_endtime = [0 for _ in range(self.n_jobs)]               #机器 的上一个工件的完工时间
        self.machine_condition = [0 for _ in range(self.n_machines)]             #对n个机器设置其初始役龄为0
        self.h1 = h1
        self.h2 = h2        #恶化效应点和失效点（h1,h2）
        self.machine_utilizition = [0 for _ in range(self.n_machines)]     #n个机器的初始利用率均为0
        self.de_factor = 0.2      #恶化系数
        self.total_cost = 0      #奖励
        self.m1_cost = 10       #小修一次所需成本
        self.m2_cost = 40  # 大修一次所需成本
        self.m1_time = 5    # 小修一次所需时间
        self.m2_time = 15   #  大修一次所需成本
        self.p_cost = 3     #单位时间机器运行成本




    def step(self, action):
        if self.done:
            print("Environment already done")
            return self.state, 0, self.done, {}

        #if action == 0 :            #第一个动作是通过 选择下一道加工时间最短的工件 再选择一个可以用的机器



        job , machine = action

        if self.job_status[job] > len(self.processing_times[0]):      #判断工件是否已经完成
            return self.state, 0, self.done, {}

        if self.machine_condition[machine] > self.h2:  # 机器状态超过h2，需要直接进行小修 后进行加工
            self.machine_condition[machine] = self.h1     # 小修
            self.total_cost = self.m1_cost           #小修成本
            self.machine_status[machine] += self.m1_time
            self.prev_job_endtime[job] += self.m1_time
            self.time = max(self.machine_status)
        #实际加工时间
        actual_processing_time = self.processing_times[job][self.job_status[job] - 1][machine] + max(0,self.de_factor * (self.machine_condition[machine] - self.h1))

        prev_process = self.job_status[job] - 1
        if prev_process == 0:        #是工件的第一道工序

            self.machine_status[machine] += actual_processing_time  #
            self.prev_job_endtime[job] = actual_processing_time
            self.time = max(self.machine_status)
            self.total_cost += self.p_cost*actual_processing_time


        if prev_process > 0:        #不是第一道工序
            if self.time == self.prev_job_endtime[job]:   #同一道工件的连续的工序
                self.machine_status[machine] = self.time+self.processing_times[job][self.job_status[job] - 1][machine]



            if self.time > self.prev_job_endtime[job]:
                self.machine_status[machine] += self.processing_times[job][self.job_status[job] - 1][machine]
            self.prev_job_endtime[job] = self.machine_status[machine]
            self.time = max(self.machine_status)

        self.job_status[job] += 1

        if all(status == (len(self.processing_times[0])+1) for status in self.job_status):         # 结束判断语句
            self.done = True

        self.state = (np.mean(self.machine_status), np.mean(self.job_status),
                      self.n_jobs - sum(np.array(self.job_status) == self.n_machines + 1))

        #print("jiqi", self.machine_status)

        return self.state, -self.total_cost, self.done, {}


    def reset(self):
        self.time = 0
        self.job_status = [1 for _ in range(self.n_jobs)]
        self.machine_status = [0 for _ in range(self.n_machines)]
        self.done = False
        self.state = (np.mean(self.machine_status), np.mean(self.job_status),
                      self.n_jobs - sum(np.array(self.job_status) == self.n_machines + 1))
        self.prev_job_endtime = [0 for _ in range(self.n_jobs)]

        return self.state

if __name__ == "__main__":

    processing_times = [[[1, 1], [1, 1],[3,3]],
                        [[1, 1], [1, 1],[3,3]],
                        [[1, 1], [1, 1],[3,3]]]

    env = FJSP(n_jobs=3, n_machines=2, processing_times=processing_times,h1=1,h2=1)
    state = env.reset()

    while not env.done:
        action = ( random.randint(0, env.n_jobs - 1),random.randint(0, env.n_machines - 1))

        print(action)
        state, reward, done, _ = env.step(action)
        # print("state:", state)
        # print("reward:", reward)
        # print("done:", done)
    print(env.machine_status)