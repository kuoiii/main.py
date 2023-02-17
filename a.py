# !/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : a.py
# Author: HuXianyong
# Date  : 2022/10/1

import tensorflow as tf
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#离散事件仿真 析取图

#数字化车间排产调度

import random

# Number of workpieces and machines
num_workpieces = 10
num_machines = 5

# List to store the processing times for each workpiece on each machine
processing_times = []
for i in range(num_workpieces):
    processing_times.append([random.randint(1, 10) for j in range(num_machines)])

# Dataset
dataset = {
    'num_workpieces': num_workpieces,
    'num_machines': num_machines,
    'processing_times': processing_times
}

print(dataset)
