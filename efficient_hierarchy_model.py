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
        pass

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
    update_coef: opinion updating coefficient, use difference in alpha if None
    speak_prob: 'non-uniform' = calculate speaking probability based on alpha and k,
                'uniform' = uniform speaking probability
    """
    def __init__(self, N, x_threshold, k, nlead, lead_alpha,
                 follw_alpha, lim_listeners, update_coef=None,
                 speak_prob='non-uniform', track_agents=False):
        self.num_agents = N
        self.x_threshold = x_threshold
        self.k = k
        self.nlead = nlead
        self.lead_alpha = lead_alpha
        self.follw_alpha = follw_alpha
        self.lim_listeners = lim_listeners
        self.update_coef = update_coef
        self.track_agents = track_agents
        self.n_event = 0
        self.schedule = RandomActivation(self)
        self.running = True

        # initialize population
        self.construct_population(self.nlead)

        # pre-compute speaking probabilities
        if speak_prob == 'non_uniform':
            self.speak_probs = self.calc_talk_prob()
        else:
            self.speak_probs = np.ones(self.num_agents)/self.num_agents

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

    def calc_talk_prob(self):
        speak_probs = np.zeros(self.num_agents)
        speak_denom = 0
        for j, agent in enumerate(self.schedule.agents):
            speak_val = np.power(agent.alpha, self.k)
            speak_probs[j] = speak_val
            speak_denom += speak_val
        speak_probs = speak_probs / speak_denom

        return speak_probs

    def mean_opinion(self):
        return np.mean([agent.opinion for agent in self.schedule.agents])

    def sd_opinion(self):
        return np.std([agent.opinion for agent in self.schedule.agents])

    # Model time step
    def step(self):

        pop_inds = np.arange(self.num_agents)

        opi_sd = self.sd_opinion()
        # print(opi_sd)
        if opi_sd < self.x_threshold:
            self.running = False

        # select speaker
        speaker_ind = np.random.choice(pop_inds, p=self.speak_probs, size=1)[0]
        speaker = self.schedule.agents[speaker_ind]

        # select listeners
        listener_inds = np.random.choice(pop_inds[pop_inds != speaker_ind],
                                         size=self.lim_listeners, replace=False)

        # activate to collect agent-level data
        if self.track_agents:
            self.schedule.step()

        # update listeners' opinion
        for ind in listener_inds:
            listener = self.schedule.agents[ind]

            # calculate opinion updating coefficient
            if self.update_coef is None:
                update_coef = speaker.alpha - listener.alpha

                if update_coef <= 0:
                    update_coef = 0.01
            else:
                update_coef = self.update_coef

            # calculate difference in opinion
            opinion_diff = speaker.opinion - listener.opinion

            # update opinion (fixed updating coefficient replacing alpha_diff)
            listener.opinion = listener.opinion + (update_coef * opinion_diff)

        self.n_event += 1