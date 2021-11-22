# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation

class OpinionAgent(Agent):
    """An agent with initial opinion"""
    def __init__(self, unique_id, opinion, model):
        super().__init__(unique_id, model)
        self.opinion = opinion

    def step(self):
        if self.role == "speaker":
            listeners = np.random.choice(self.schedule.agents, self.lim_listeners, replace=False)
            for listener in listeners:
                opinion_diff = self.opinion - listener.opinion
                listener.opinion = listener.opinion + (0.01*opinion_diff)

class OpinionFormationModel(Model):
    """A model with some number of agents"""
    def __init__(self, N, lim_listeners):
        self.num_agents = N
        self.lim_listeners = lim_listeners
        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            opinion = np.random.uniform(0, 1, size=1)
            a = OpinionAgent(i, opinion, self)
            self.schedule.add(a)

    def step(self):
        speaker = np.random.choice(self.schedule.agents, size=1)
        self.schedule.step()
