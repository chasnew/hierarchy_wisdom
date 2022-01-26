# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

class OpinionAgent():
    """
    An agent with initial opinion and influence

    unique_id: agent's unique id
    alpha: agent's influence that's translated into probability of speaking
    opinion: agent's opinion
    """
    def __init__(self, alpha, opinion, w=0):
        self.alpha = alpha # influence
        self.opinion = opinion
        self.w = w

    def __lt__(self, agent):
        return self.alpha < agent.alpha

class Community(Agent):
    """
    A community of opinion agents that try to reach a consensus to produce payoffs

    N: the number of agents in the community
    x_threshold: consensus threshold
    k: exponent determining the mapping between influence and speaking probability
    lim_listeners: number of listeners per speaking event
    """
    def __init__(self, unique_id, model, N, x_threshold, k, lim_listeners):
        super().__init__(unique_id, model)

        self.N = N
        self.x_thresh = x_threshold
        self.k = k
        self.lim_listeners = lim_listeners
        self.n_event = 0

        # create a population of agents
        self.population = self.create_population()

        # pre-compute speaking probabilities
        self.speak_probs = self.calc_talk_prob()

    def create_population(self):
        population = []
        for i in range(self.N):
            population.append(OpinionAgent(alpha=np.random.rand(),
                                           opinion=np.random.rand()))

        return population

    def calc_talk_prob(self):
        speak_probs = np.zeros(self.N)
        speak_denom = 0
        for j, agent in enumerate(self.population):
            speak_val = np.power(agent.alpha, self.k)
            speak_probs[j] = speak_val
            speak_denom += speak_val
        speak_probs = speak_probs / speak_denom

        return speak_probs

    def mean_opinion(self):
        return np.mean([agent.opinion for agent in self.population])

    def sd_opinion(self):
        return np.std([agent.opinion for agent in self.population])

    def step(self):

        pop_inds = np.arange(len(self.population))
        opi_sd = self.sd_opinion()

        self.n_event = 0

        while(opi_sd < self.x_threshold):

            # select speaker
            speaker_ind = np.random.choice(pop_inds, p=self.speak_probs, size=1)[0]
            speaker = self.schedule.agents[speaker_ind]

            # select listeners
            listener_inds = np.random.choice(pop_inds[pop_inds != speaker_ind],
                                             size=self.lim_listeners, replace=False)

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



class EvoOpinionModel(Model):
    """
    A consensus building model with opinion agents that have different levels of influence

    init_n: initial number of agents in each community
    x_threshold: consensus threshold
    k: exponent determining the mapping between influence and speaking probability
    lim_listeners: number of listeners per speaking event
    np: number of community patches
    mu: mutation rate of influence value
    mu_var: variance of mutation rate
    K: carrying capacity for population growth equation
    ra: intrinsic growth rate
    gammar: steepness of growth rate induced by extra resources
    betar: max increase in growth rate induced by extra resources
    gammab: steepness of increase in benefit induced by participants
    betab: max increase in benefit induced by number of participants
    S: the benefit that will inherit to the next generation
    b_mid: group size at the sigmoid's midpoint (sigmoid parameter)
    Ct: time constraints on group consensus building
    d: ecological inequality
    """
    def __init__(self, init_n, x_threshold, k, lead_alpha, follw_alpha, lim_listeners,
                 np, mu, mu_var, K, ra, gammar, betar, gammab, betab, S, b_mid, Ct, d,
                 nlead=0, random_leadx=True):
        self.init_n = init_n
        self.x_threshold = x_threshold
        self.k = k
        self.lim_listeners = lim_listeners
        self.np = np
        self.mu = mu
        self.mu_var = mu_var
        self.K = K
        self.ra = ra
        self.gammar = gammar
        self.betar = betar
        self.gammab = gammab
        self.betab = betab
        self.S = S
        self.b_mid = b_mid
        self.Ct = Ct
        self.d = d
        self.schedule = RandomActivation(self)
        self.running = True

        # initialize communities
        for i in range(np):
            self.schedule.add(Community(self.init_n, self.x_threshold,
                                        self.k, self.lim_listeners))

        # # Define data collectors
        # self.datacollector = DataCollector(
        #     model_reporters={'mean_opinion': self.mean_opinion,
        #                      'sd_opinion': self.sd_opinion,
        #                      'n_event': 'n_event'},
        #     agent_reporters={'opinion': 'opinion'}
        # )

    # Model time step
    def step(self):
        self.datacollector.collect(self)

        Bt = np.zeros(self.np) # group additional benefit induced by more participants

        # activate consensus building for each community
        self.schedule.step()