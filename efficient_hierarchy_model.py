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
    def __init__(self, unique_id, alpha, opinion, model, w=0):
        super().__init__(unique_id, model)
        self.alpha = alpha # influence
        self.opinion = opinion
        self.w = w

    def __lt__(self, agent):
        return self.alpha < agent.alpha

    def step(self):
        # activate to collect data
        None

class OpinionModel(Model):
    """
    A consensus building model with opinion agents that have different levels of influence

    N: number of agents
    x_threshold: threshold of standard variation of opinions under which the consensus-building is complete
    k: exponent determining the mapping between influence and speaking probability
    nlead: number of leaders (agents w/ high influence)
    lead_alpha: leaders' influence
    follw_alpha: followers' influence
    lim_listeners: number of listeners
    """
    def __init__(self, N, x_threshold, k, nlead, lead_alpha, follw_alpha, lim_listeners):
        self.num_agents = N
        self.x_threshold = x_threshold
        self.k = k
        self.nlead = nlead
        self.lead_alpha = lead_alpha
        self.follw_alpha = follw_alpha
        self.lim_listeners = lim_listeners
        self.n_event = 0
        self.schedule = RandomActivation(self)
        self.running = True

        # initialize population
        self.construct_population(self.nlead)

        # Define data collectors
        self.datacollector = DataCollector(
            model_reporters={'mean_opinion': self.mean_opinion,
                             'sd_opinion': self.sd_opinion,
                             'n_event': 'n_event'},
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

        # compute speaking probability
        speak_probs = np.zeros(self.num_agents)
        speak_denom = 0
        for j, agent in enumerate(self.schedule.agents):
            speak_val = np.power(agent.alpha, self.k)
            speak_probs[j] = speak_val
            speak_denom += speak_val
        speak_probs = speak_probs / speak_denom

        pop_inds = np.arange(self.num_agents)

        opi_sd = self.sd_opinion()
        # print(opi_sd)
        if opi_sd < self.x_threshold:
            self.running = False

        # select speaker
        speaker_ind = np.random.choice(pop_inds, p=speak_probs, size=1)[0]
        speaker = self.schedule.agents[speaker_ind]

        # select listeners
        listener_inds = np.random.choice(pop_inds[pop_inds != speaker_ind],
                                         size=self.lim_listeners, replace=False)

        # self.schedule.step() # activate to collect agent data

        # update listeners' opinion
        for ind in listener_inds:
            listener = self.schedule.agents[ind]

            # calculate difference in influence
            alpha_diff = speaker.alpha - listener.alpha
            if alpha_diff <= 0:
                alpha_diff = 0.01

            # calculate difference in opinion
            opinion_diff = speaker.opinion - listener.opinion

            # update opinion
            listener.opinion = listener.opinion + (alpha_diff * opinion_diff)

        self.n_event += 1