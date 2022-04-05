# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np
from scipy.stats import truncnorm

import multiprocessing as mp

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

class Community():
    """
    A community of opinion agents that try to reach a consensus to produce payoffs

    N: the number of agents in the community
    x_threshold: consensus threshold
    k: exponent determining the mapping between influence and speaking probability
    lim_listeners: number of listeners per speaking event
    mu_rate: mutation rate of influence value
    alpha_var: variance of influence when mutation occurs
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
    def __init__(self, unique_id, model, N, x_threshold, k, lim_listeners,
                 mu_rate, alpha_var, K, ra, gammar, betar, gammab, betab, S, b_mid, Ct, d):
        self.id = unique_id
        self.N = N
        self.x_threshold = x_threshold
        self.k = k
        self.lim_listeners = lim_listeners
        self.mu_rate = mu_rate
        self.alpha_var = alpha_var
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
        self.n_event = 0
        self.Bt = 0 # additional resource produced by group

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

    def reproduce(self):

        offsprings = []

        for agent in self.population:
            # might need to adjust opinion to inheritance mode
            o_n = np.random.poisson(agent.w)

            # randomize mutation
            mutation_masks = np.random.choice([0,1], p=[1-self.mu_rate, self.mu_rate], size=o_n)

            # adjusted truncated thresholds
            a, b = (0 - agent.opinion) / self.alpha_var, (1 - agent.opinion) / self.alpha_var

            # mutated alpha values
            o_alphas = truncnorm.rvs(a , b, loc=agent.alpha, scale=self.alpha_var, size=o_n)
            o_alphas = [agent.alpha if mutation_masks[i] else o_alphas[i] for i in range(o_n)]

            offspring_tmp = [OpinionAgent(alpha=o_alphas[i],
                                          opinion=np.random.rand())
                             for i in range(o_n)]
            offsprings.extend(offspring_tmp)

        # reproduction step
        self.population = offsprings

        # update population parameter
        self.N = len(self.population)
        self.speak_probs = self.calc_talk_prob()

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

    def mean_fitness(self):
        return np.mean([agent.w for agent in self.population])

    def mean_influence(self):
        return np.mean([agent.alpha for agent in self.population])

    # Pearson’s moment coefficient of skewness
    def influence_skewness(self):
        alpha_array = np.array([agent.alpha for agent in self.population])
        alpha_devs = alpha_array - np.mean(alpha_array)

        return np.mean(np.power(alpha_devs, 3))/np.power(np.std(alpha_array), 3)

    def make_decision(self):

        pop_inds = np.arange(len(self.population))
        opi_sd = self.sd_opinion()
        init_opinions = np.array([agent.opinion for agent in self.population])

        self.n_event = 0

        while(opi_sd > self.x_threshold):

            # select speaker
            speaker_ind = np.random.choice(pop_inds, p=self.speak_probs, size=1)[0]
            speaker = self.population[speaker_ind]

            # select listeners
            listener_inds = np.random.choice(pop_inds[pop_inds != speaker_ind],
                                             size=self.lim_listeners, replace=False)

            # update listeners' opinion
            for ind in listener_inds:
                listener = self.population[ind]

                # calculate difference in influence
                alpha_diff = speaker.alpha - listener.alpha
                if alpha_diff <= 0:
                    alpha_diff = 0.01

                # calculate difference in opinion
                opinion_diff = speaker.opinion - listener.opinion

                # update opinion
                listener.opinion = listener.opinion + (alpha_diff * opinion_diff)

            opi_sd = self.sd_opinion()
            self.n_event += 1

        # Calculate additional resource produced by the group (group-level payoff)
        self.Bt = (self.Bt * self.S) +\
                  (self.betab/(1 + np.exp(-self.gammab*(self.N - self.b_mid)))) -\
                  (self.Ct * self.n_event)

        # Calculate the share of resources each individual receives
        alphar = 1 - np.abs(init_opinions - self.mean_opinion())

        pt = (1 + (self.d * alphar))
        pt = pt/np.sum(pt)

        # Calculate additional growth rate for each individual
        rb = self.betar * (1 - np.exp(-self.gammar * (self.Bt * pt)))

        # Calculate fitness for each individual
        w = (self.ra / (1 + (self.N/self.K))) + rb
        w[ w < 0 ] = 0

        for i, agent in enumerate(self.population):
            agent.w = w[i]


class EvoOpinionModel():
    """
    A consensus building model with opinion agents that have different levels of influence

    init_n: initial number of agents in each community
    x_threshold: consensus threshold
    k: exponent determining the mapping between influence and speaking probability
    lim_listeners: number of listeners per speaking event
    np: number of community patches
    mu_rate: mutation rate of influence value
    alpha_var: variance of influence when mutation occurs
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
    m: migration rate
    """
    def __init__(self, init_n, x_threshold, k, lim_listeners, np, mu_rate, alpha_var,
                 K, ra, gammar, betar, gammab, betab, S, b_mid, Ct, d, m):
        self.init_n = init_n
        self.x_threshold = x_threshold
        self.k = k
        self.lim_listeners = lim_listeners
        self.np = np
        self.mu_rate = mu_rate
        self.alpha_var = alpha_var
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
        self.m = m
        self.step_count = 0
        self.communities = set() # or list()
        self.agent_reporter = {'group_id': lambda c: c.id,
                               'n_event': lambda c: c.n_event,
                               'group_size': lambda c: c.N,
                               'extra_resource': lambda c: c.Bt,
                               'avg_alpha': lambda c: c.mean_influence(),
                               'alpha_skewness': lambda c: c.influence_skewness()}
        self.datacollector = {key: [] for key in self.agent_reporter.keys()}
        self.datacollector['step'] = []

        # initialize communities
        for i in range(self.np):
            self.communities.add(Community(unique_id=i,
                                           model=self,
                                           N=self.init_n,
                                           x_threshold=self.x_threshold,
                                           k=self.k,
                                           lim_listeners=self.lim_listeners,
                                           mu_rate=self.mu_rate,
                                           alpha_var=self.alpha_var,
                                           K=self.K,
                                           ra=self.ra,
                                           gammar=self.gammar,
                                           betar=self.betar,
                                           gammab=self.gammab,
                                           betab=self.betab,
                                           S=self.S,
                                           b_mid=self.b_mid,
                                           Ct=self.Ct,
                                           d=self.d))

    # Model time step
    def step(self, verbose=False, process_num=1):

        # initiate multicore-processing pool
        if process_num == -1:
            process_num = mp.cpu_count()
            pool = mp.Pool(processes=process_num)
        elif process_num > 1:
            pool = mp.Pool(processes=process_num)

        # activate consensus building for each community
        if process_num == 1:
            for community in self.communities:
                if verbose:
                    print('activating community', community.id)
                community.make_decision()
        else:
            for community in self.communities:
                pool.apply_async(community.make_decision())

        self.step_count += 1

        # collecting model results
        for key, collect_func in self.agent_reporter.items():
            self.datacollector[key].extend(list(map(collect_func, self.communities)))

        self.datacollector['step'].extend([self.step_count] * self.np)

        # Reproduction step
        if process_num == 1:
            for community in self.communities:
                community.reproduce()
        else:
            for community in self.communities:
                pool.apply_async(community.reproduce())

        if process_num > 1:
            pool.close()

        # Migration step
        if self.np > 1 and self.m is not None:
            c_migrants = []

            ## sample migrants
            for i in range(self.np):
                community = self.communities[i]
                pop_ids = np.arange(len(community.population))
                m_masks = np.random.binomial(size=pop_ids.shape[0], n=1, p=self.m)
                m_ids = pop_ids[m_masks == 1]
                c_migrants.append(community.population.pop(m_id) for m_id in reversed(m_ids))

            ## assign patches
            c_ids = np.arange(self.np)
            for i in range(self.np):
                new_c_ids = np.random.choice(c_ids[c_ids != i], size=len(c_migrants[i]), replace=True)

                for c_id in new_c_ids:
                    self.communities[c_id].population.append(c_migrants[i].pop(0))