# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

class OpinionAgent(Agent):
    """
    An agent with initial opinion and influence

    unique_id: agent's unique id
    alpha: agent's influence that's translated into probability of speaking
    opinion: agent's opinion
    """
    def __init__(self, unique_id, alpha, opinion, model):
        super().__init__(unique_id, model)
        self.alpha = alpha # influence
        self.opinion = opinion
        self.role = None

    def step(self):
        print(self.unique_id)

class OpinionModel(Model):
    """A model with opinion agents"""
    def __init__(self, N, x_threshold, k, nlead, lead_alpha, follw_alpha, lim_listeners):
        self.num_agents = N
        self.x_threshold = x_threshold # consensus threshold
        self.k = k # exponent determining the mapping between influence and talk prob
        self.nlead = nlead # number of leaders
        self.lead_alpha = lead_alpha # leader's influence
        self.follw_alpha = follw_alpha # followers' influence
        self.lim_listeners = lim_listeners # number of listeners
        self.schedule = RandomActivation(self)
        # self.running = True

        # initialize population
        self.construct_population(self.nlead)

        # Define data collectors
        self.datacollector = DataCollector(
            model_reporters={'mean_opinion': self.mean_opinion,
                             'sd_opinion': self.sd_opinion},
            agent_reporters={'opinion': 'opinion'}
        )

    def construct_population(self, nlead, random_leadx=True):
        x_max = 1

        # initialize leaders
        for i in range(nlead):
            if random_leadx:
                # randomize leaders' opinion [0,1]
                a = OpinionAgent(i, self.lead_alpha, np.random.rand()*x_max, self)
                self.schedule.add(a)
            else:
                # evenly spaced leaders' opinion
                tmp_opinion = (i+1)/(nlead+1)
                a = OpinionAgent(i, self.lead_alpha, tmp_opinion, self)
                self.schedule.add(a)

        # initialize followers
        for i in range(nlead, self.num_agents):
            opinion = np.random.uniform(0, 1, size=1)
            a = OpinionAgent(i, self.follw_alpha, opinion, self)
            self.schedule.add(a)

    def mean_opinion(self):
        return np.mean([agent.opinion for agent in self.schedule.agents])

    def sd_opinion(self):
        return np.std([agent.opinion for agent in self.schedule.agents])

    # Model time step
    def step(self):
        self.datacollector.collect(self)

        # select speaker and listeners
        speak_probs = np.zeros(self.num_agents)
        speak_denom = 0
        for j, agent in enumerate(self.schedule.agents):
            speak_val = np.power(agent.alpha, self.k)
            speak_probs[j] = speak_val
            speak_denom += speak_val
        speak_probs = speak_probs / speak_denom

        pop_inds = np.arange(self.num_agents)

        # make consensus
        speaker_ind = np.random.choice(pop_inds, p=speak_probs, size=1)[0]
        speaker = self.schedule.agents[speaker_ind]

        listener_inds = np.random.choice(pop_inds[pop_inds != speaker_ind],
                                         size=self.lim_listeners, replace=False)

        # update listeners' opinion
        for ind in listener_inds:
            listener = self.schedule.agents[ind]
            print('listener id = ', listener.unique_id)
            opinion_diff = speaker.opinion - listener.opinion
            listener.opinion = listener.opinion + (0.01 * opinion_diff)
            print('listener opinion = ', listener.opinion)