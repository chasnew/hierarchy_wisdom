# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np
from scipy.stats import truncnorm
import pickle
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

    unique_id: ID of the community
    model: evolutionary model super class that stores all parameters
    N: the number of agents in the community
    """
    def __init__(self, unique_id, model, N):
        self.id = unique_id
        self.model = model
        self.N = N
        self.n_event = 0
        self.Bt = 0 # additional resource produced by group

        # create a population of agents
        self.population = self.create_population()

    def create_population(self):
        population = []
        for i in range(self.N):
            population.append(OpinionAgent(alpha=np.random.rand(),
                                           opinion=np.random.rand()))

        return population

    def reproduce(self):
        np.random.seed()

        offsprings = []

        for agent in self.population:
            # might need to adjust opinion to inheritance mode
            o_n = np.random.poisson(agent.w)

            # randomize mutation
            mutation_masks = np.random.choice([0,1], p=[1-self.model.mu_rate, self.model.mu_rate], size=o_n)

            # adjusted truncated thresholds
            a, b = (0 - agent.opinion) / self.model.alpha_var, (1 - agent.opinion) / self.model.alpha_var

            # mutated alpha values
            o_alphas = truncnorm.rvs(a , b, loc=agent.alpha, scale=self.model.alpha_var, size=o_n)
            o_alphas = [agent.alpha if mutation_masks[i] == 0 else o_alphas[i] for i in range(o_n)]

            offspring_tmp = [OpinionAgent(alpha=o_alphas[i],
                                          opinion=np.random.rand())
                             for i in range(o_n)]
            offsprings.extend(offspring_tmp)

        # reproduction step
        self.population = offsprings

        # update population parameter
        self.N = len(self.population)

    def calc_talk_prob(self):
        speak_probs = np.zeros(self.N)
        speak_denom = 0
        for j, agent in enumerate(self.population):
            speak_val = np.power(agent.alpha, self.model.k)
            speak_probs[j] = speak_val
            speak_denom += speak_val
        speak_probs = speak_probs / speak_denom

        return speak_probs

    def mean_opinion(self):
        return np.mean([agent.opinion for agent in self.population])

    def sd_opinion(self):
        return np.std([agent.opinion for agent in self.population])

    def left_sd_opinion(self):
        return np.std([agent.opinion for agent in self.population if agent.opinion <= 0.5])

    def right_sd_opinion(self):
        return np.std([agent.opinion for agent in self.population if agent.opinion > 0.5])

    def left_dissent(self):
        return np.mean(np.abs([0.5 - agent.opinion for agent in self.population if agent.opinion <= 0.5]))

    def right_dissent(self):
        return np.mean(np.abs([0.5 - agent.opinion for agent in self.population if agent.opinion > 0.5]))

    def choice_prop(self):
        opi_array = np.array([agent.opinion for agent in self.population])
        return np.mean(opi_array < 0.5)

    def majority_side(self):
        if self.choice_prop() > self.model.x_threshold:
            return 'left'
        else:
            return 'right'

    def mean_fitness(self):
        return np.mean([agent.w for agent in self.population])

    def mean_influence(self):
        return np.mean([agent.alpha for agent in self.population])

    def sd_influence(self):
        return np.std([agent.alpha for agent in self.population])

    # Pearson’s moment coefficient of skewness
    # Function in original Java code: skewness = [n / (n -1) (n - 2)] sum[(x_i - mean)^3] / std^3
    def influence_skewness(self):
        alpha_array = np.array([agent.alpha for agent in self.population])
        alpha_devs = alpha_array - np.mean(alpha_array)

        return np.mean(np.power(alpha_devs, 3))/np.power(np.std(alpha_array), 3)

    def check_consensus(self, criterion='sd_threshold'):
        if criterion == 'sd_threshold':
            opi_sd = self.sd_opinion()
            return (opi_sd < self.model.x_threshold)
        elif criterion == 'prop_threshold':
            c_prop = self.choice_prop()
            consensus_mask = (c_prop < 1 - self.model.x_threshold) | (c_prop > self.model.x_threshold)
            return (consensus_mask)
        else:
            return (True)

    def make_decision(self):
        np.random.seed()

        self.speak_probs = self.calc_talk_prob()

        pop_inds = np.arange(len(self.population))
        init_opinions = np.array([agent.opinion for agent in self.population])

        self.n_event = 0

        while(not self.check_consensus(criterion=self.model.criterion)):

            # select speaker
            speaker_inds = np.random.choice(pop_inds, p=self.speak_probs,
                                            size=self.model.lim_speakers, replace=False)
            if self.model.lim_speakers > 1:
                speakers = [self.population[i] for i in speaker_inds]

                speaker_alphas = []
                speaker_opinions = []
                for i in range(len(speakers)):
                     speaker_alphas.append(speakers[i].alpha)
                     speaker_opinions.append(speakers[i].opinion)

                speaker_alphas = np.array(speaker_alphas)
                speaker_opinions = np.array(speaker_opinions)

                speaker_influence = np.mean(speaker_alphas)
                speaker_position = np.sum((speaker_alphas / np.sum(speaker_alphas)) * speaker_opinions)
            else:
                speaker_ind = speaker_inds[0]
                speaker_influence = self.population[speaker_ind].alpha
                speaker_position = self.population[speaker_ind].opinion

            non_speaker_masks = np.ones(len(self.population), np.bool)
            non_speaker_masks[speaker_inds] = 0

            # select listeners
            listener_inds = np.random.choice(pop_inds[non_speaker_masks],
                                             size=self.model.lim_listeners, replace=False)

            # update listeners' opinion
            for ind in listener_inds:
                listener = self.population[ind]

                # calculate difference in influence
                alpha_diff = speaker_influence - listener.alpha
                if alpha_diff <= 0:
                    alpha_diff = 0.01

                # calculate difference in opinion
                opinion_diff = speaker_position - listener.opinion

                # update opinion
                listener.opinion = listener.opinion + (alpha_diff * opinion_diff)

            opi_sd = self.sd_opinion()
            self.n_event += 1

        # Calculate additional resource produced by the group (group-level payoff)
        prev_Bt = self.Bt
        self.Bt = (self.model.betab/(1 + np.exp(-self.model.gammab*(self.N - self.model.b_mid)))) -\
                  self.model.Ct * (self.n_event ** self.model.SAt)
        self.Bt = max(0, self.Bt)

        # Calculate the share of resources each individual receives
        alphar = 1 - np.abs(init_opinions - self.mean_opinion())

        pt = (1 + (self.model.d * alphar))
        pt = pt/np.sum(pt)

        # Calculate additional growth rate for each individual
        rb = self.model.betar * (1 - np.exp(-self.model.gammar * ((self.Bt + (self.model.S * prev_Bt)) * pt)))

        # Calculate fitness for each individual
        w = (self.model.ra / (1 + (self.N/self.model.K))) + rb
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
    SAt: speed-accuracy tradeoff parameter
        where 1 will favor fast consensus and -1 will favor slow consensus
    d: ecological inequality
    m: migration rate
    load_communities: a set of communities provided to the model in a scenario
        where user wants to resume the simulation
    """
    def __init__(self, init_n, x_threshold, k, lim_speakers, lim_listeners, np, mu_rate, alpha_var,
                 K, ra, gammar, betar, gammab, betab, S, b_mid, Ct, SAt, d, m,
                 criterion='sd_threshold', load_communities=None):
        self.init_n = init_n
        self.x_threshold = x_threshold
        self.k = k
        self.lim_speakers = lim_speakers
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
        self.SAt = SAt
        self.d = d
        self.m = m
        self.criterion = criterion
        self.step_count = 0
        self.communities = set() # or list()
        self.agent_reporter = {'group_id': lambda c: c.id,
                               'n_event': lambda c: c.n_event,
                               'group_size': lambda c: c.N,
                               # 'extra_resource': lambda c: c.Bt,
                               # 'avg_fitness': lambda c: c.mean_fitness(),
                               'avg_alpha': lambda c: c.mean_influence(),
                               'alpha_skewness': lambda c: c.influence_skewness()}
        if criterion == 'prop_threshold':
            self.agent_reporter.update({'sd_opinion': lambda c: c.sd_opinion(),
                                        'left_sd_opinion': lambda c: c.left_sd_opinion(),
                                        'right_sd_opinion': lambda c: c.right_sd_opinion(),
                                        'left_dissent': lambda c: c.left_dissent(),
                                        'right_dissent': lambda c: c.right_dissent(),
                                        'majority_side': lambda c: c.majority_side()})
        self.datacollector = {key: [] for key in self.agent_reporter.keys()}
        self.datacollector['step'] = []

        # initialize communities
        if load_communities is not None:
            self.communities = load_communities
        else:
            for i in range(self.np):
                self.communities.add(Community(unique_id=i,
                                               model=self,
                                               N=self.init_n))

    # Model time step
    def step(self, verbose=False, process_num=1):

        # initiate multicore-processing pool
        if process_num == -1:
            process_num = mp.cpu_count()

        if process_num > 1:
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

            pool.close()
            pool.join()

        self.step_count += 1

        # collecting model results
        for key, collect_func in self.agent_reporter.items():
            self.datacollector[key].extend(list(map(collect_func, self.communities)))

        self.datacollector['step'].extend([self.step_count] * self.np)

        if process_num > 1:
            pool = mp.Pool(processes=process_num)

        # Reproduction step
        if process_num == 1:
            for community in self.communities:
                community.reproduce()
        else:
            for community in self.communities:
                pool.apply_async(community.reproduce())

            pool.close()
            pool.join()

        # for c in self.communities:
        #     print('community', c.id, ': ', c.N)

        # Migration step
        if self.np > 1 and self.m is not None:
            list_com = list(self.communities)
            c_migrants = []

            ## sample migrants
            for i in range(self.np):
                community = list_com[i]
                pop_ids = np.arange(len(community.population))
                m_masks = np.random.binomial(size=pop_ids.shape[0], n=1, p=self.m)
                m_ids = pop_ids[m_masks == 1]
                c_migrants.append([community.population.pop(m_id) for m_id in reversed(m_ids)])

            ## assign patches
            c_ids = np.arange(self.np)
            for i in range(self.np):
                new_c_ids = np.random.choice(c_ids[c_ids != i], size=len(c_migrants[i]), replace=True)

                for c_id in new_c_ids:
                    list_com[c_id].population.append(c_migrants[i].pop(0))

            ## update group size
            for c in self.communities:
                c.N = len(c.population)

        # for c in self.communities:
        #     print('community', c.id, ': ', c.N)

    def save_communities(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.communities, file)