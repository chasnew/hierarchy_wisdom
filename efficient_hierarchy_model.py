# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

class OpinionAgent(Agent):
    """An agent with initial opinion"""
    def __init__(self, unique_id, opinion, model):
        super().__init__(unique_id, model)
        self.opinion = opinion
        self.role = None

    def step(self):
        print(self.unique_id)

class OpinionModel(Model):
    """A model with some number of agents"""
    def __init__(self, N, lim_listeners):
        self.num_agents = N
        self.lim_listeners = lim_listeners
        self.schedule = RandomActivation(self)
        # self.running = True

        # Create agents
        for i in range(self.num_agents):
            opinion = np.random.uniform(0, 1, size=1)
            a = OpinionAgent(i, opinion, self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={'mean_opinion': self.mean_opinion,
                             'sd_opinion': self.sd_opinion}
            agent_reporters={'opinion': 'opinion'}
        )

    def mean_opinion(self):
        return np.mean([agent.opinion for agent in self.schedule.agents])

    def sd_opinion(self):
        return np.std([agent.opinion for agent in self.schedule.agents])

    def step(self):
        self.datacollector.collect(self)

        # select speaker and listeners
        tmp_inds = np.arange(self.num_agents)

        speaker_ind = np.random.choice(tmp_inds, size=1)[0]
        speaker = self.schedule.agents[speaker_ind]

        listener_inds = np.random.choice(tmp_inds[tmp_inds != speaker_ind],
                                         size=self.lim_listeners, replace=False)

        # update listeners' opinion
        for ind in listener_inds:
            listener = self.schedule.agents[ind]
            print('listener id = ', listener.unique_id)
            opinion_diff = speaker.opinion - listener.opinion
            listener.opinion = listener.opinion + (0.01 * opinion_diff)
            print('listener opinion = ', listener.opinion)