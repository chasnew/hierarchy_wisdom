# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np

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

class OpinionModel():
    """
    A consensus building model with opinion agents that have different levels of influence

    N: number of agents
    x_threshold:
        if criterion = 'sd_threshold' (default),
        this value indicates the standard deviation of opinions under which the consensus-building is complete
        if criterion = 'prop_threshold',
        this value indicates the required proportion of opinions to fall on one side of the opinion continuum
        if criterio = 'faction',
        to be implemented
    k: exponent determining the mapping between influence and speaking probability
    nlead: number of leaders (agents w/ high influence)
    lead_alpha: leaders' influence
    follw_alpha: followers' influence
    lim_listeners: number of listeners
    criterion: consensus criterion (default = 'sd_threshold')
    update_coef: opinion updating coefficient, use difference in alpha if None
    speak_prob: 'non-uniform' = calculate speaking probability based on alpha and k,
                'uniform' = uniform speaking probability
    """
    def __init__(self, N, x_threshold, k, nlead, lead_alpha,
                 follw_alpha, lim_listeners, criterion='sd_threshold',
                 update_coef=None, speak_prob='non-uniform', track_agents=False):
        self.N = N
        self.x_threshold = x_threshold
        self.k = k
        self.nlead = nlead
        self.lead_alpha = lead_alpha
        self.follw_alpha = follw_alpha
        self.lim_listeners = lim_listeners
        self.criterion = criterion
        self.update_coef = update_coef
        self.track_agents = track_agents
        self.n_event = 0
        self.running = True

        self.model_reporter = {'mean_opinion': lambda m: m.mean_opinion(),
                               'sd_opinion': lambda m: m.sd_opinion(),
                               'N': lambda m: m.N,
                               'nlead': lambda m: m.nlead,
                               'n_event': lambda m: m.n_event}

        # initialize population
        self.population = self.create_population(self.nlead)

        # pre-compute speaking probabilities
        if speak_prob == 'non_uniform':
            self.speak_probs = self.calc_talk_prob()
        else:
            self.speak_probs = np.ones(self.N)/self.N

    def create_population(self, nlead=None, random_leadx=True):
        population = []

        if nlead is not None:
            # initialize leaders
            for i in range(nlead):
                if random_leadx:
                    # randomize leaders' opinion [0,1]
                    a = OpinionAgent(self.lead_alpha, np.random.rand())
                    population.append(a)
                else:
                    # evenly spaced leaders' opinion
                    tmp_opinion = (i + 1) / (nlead + 1)
                    a = OpinionAgent(self.lead_alpha, tmp_opinion)
                    population.append(a)

            # initialize followers
            for i in range(nlead, self.N):
                opinion = np.random.rand()  # uniform distribution [0,1]
                a = OpinionAgent(self.follw_alpha, opinion)
                population.append(a)
        else:
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

    def check_consensus(self, criterion='sd_threshold'):
        if criterion == 'sd_threshold':
            opi_sd = self.sd_opinion()
            return(opi_sd < self.x_threshold)
        elif criterion == 'prop_threshold':
            c_prop = self.choice_prop()
            consensus_mask = (c_prop < 1 - self.x_threshold) | (c_prop > self.x_threshold)
            return(consensus_mask)
        elif criterion == 'faction':
            opinion_array = np.array([agent.opinion for agent in self.population])

            return(True)

    def mean_opinion(self):
        return np.mean([agent.opinion for agent in self.population])

    def sd_opinion(self):
        return np.std([agent.opinion for agent in self.population])

    def choice_prop(self):
        opi_array = np.array([agent.opinion for agent in self.population])
        return np.mean(opi_array < 0.5)

    def report_model(self, keys):
        datacollector = {}
        for key in keys:
            datacollector[key] = self.model_reporter[key](self)

        return datacollector

    # Model time step
    def step(self):

        pop_inds = np.arange(self.N)

        # select speaker
        speaker_ind = np.random.choice(pop_inds, p=self.speak_probs, size=1)[0]
        speaker = self.population[speaker_ind]

        # select listeners
        listener_inds = np.random.choice(pop_inds[pop_inds != speaker_ind],
                                         size=self.lim_listeners, replace=False)

        # update listeners' opinion
        for ind in listener_inds:
            listener = self.population[ind]

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

        consensus = self.check_consensus(self.criterion)
        if consensus:
            self.running = False