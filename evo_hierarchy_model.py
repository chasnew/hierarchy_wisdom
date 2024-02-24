# Replication of efficient hierarchy model from https://doi.org/10.1098/rspb.2020.0693

import numpy as np
from scipy.stats import truncnorm
from collections import Counter
import joblib
import multiprocessing as mp

class OpinionAgent():
    """
    An agent with initial opinion and influence

    unique_id: agent's unique id
    alpha: agent's influence that's translated into probability of speaking
    opinion: agent's opinion
    """
    def __init__(self, alpha, opinion, w=0, cd=0):
        self.alpha = alpha # influence
        self.opinion = opinion
        self.w = w
        self.con_dist = cd

    def __lt__(self, agent):
        return self.alpha < agent.alpha

class Community():
    """
    A community of opinion agents that try to reach a consensus to produce payoffs

    unique_id: ID of the community
    model: evolutionary model super class that stores all parameters
    N: the number of agents in the community
    """
    def __init__(self, unique_id, N, x_threshold, k, lim_speakers, lim_listeners,
                 mu_rate, alpha_var, K, ra, gammar, betar, gammab, betab, S,
                 b_mid, Ct, d, criterion, init_cond):
        self.id = unique_id
        self.N = N
        self.x_threshold = x_threshold
        self.k = k
        self.lim_speakers = lim_speakers
        self.lim_listeners = lim_listeners
        self.mu_rate = mu_rate
        self.alpha_std = np.sqrt(alpha_var)
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
        self.criterion = criterion

        self.n_event = 0
        self.Bt = 0 # additional resource produced by group

        # create a population of agents
        self.population = self.create_population(init_cond)

    def create_population(self, init_cond='uniform'):
        population = []

        if init_cond == 'uniform':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.rand(),
                                               opinion=np.random.rand()))
        elif init_cond == 'all_leaders':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=1,
                                               opinion=np.random.rand()))
        elif init_cond == 'all_followers':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=0,
                                               opinion=np.random.rand()))
        elif init_cond == 'left_skew':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.beta(a=2,b=1,size=1)[0],
                                               opinion=np.random.rand()))
        elif init_cond == 'right_skew':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.beta(a=1,b=2,size=1)[0],
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
        if self.choice_prop() > self.x_threshold:
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
            return (opi_sd < self.x_threshold)
        elif criterion == 'prop_threshold':
            c_prop = self.choice_prop()
            threshold = 0.5 + np.exp(-self.x_threshold*self.n_event)/2
            consensus_mask = (c_prop < 1 - threshold) | (c_prop > threshold)
            return (consensus_mask)
        else:
            return (True)

    def make_decision(self):
        np.random.seed()

        self.speak_probs = self.calc_talk_prob()

        pop_inds = np.arange(len(self.population))
        init_opinions = np.array([agent.opinion for agent in self.population])

        self.n_event = 0

        while(not self.check_consensus(criterion=self.criterion)):

            speaker_num = self.lim_speakers

            # if number of speakers and listeners combined is bigger than the population
            if (self.lim_speakers + self.lim_listeners) > len(self.population):
                speaker_num = len(self.population) - self.lim_listeners

            # select speaker(s)
            speaker_inds = np.random.choice(pop_inds, p=self.speak_probs,
                                            size=speaker_num, replace=False)

            # average speaker opinions and influence of multiple speakers
            if speaker_num > 1:
                speakers = [self.population[i] for i in speaker_inds]

                speaker_alphas = []
                speaker_opinions = []
                for i in range(len(speakers)):
                     speaker_alphas.append(speakers[i].alpha)
                     speaker_opinions.append(speakers[i].opinion)

                speaker_alphas = np.array(speaker_alphas)
                speaker_opinions = np.array(speaker_opinions)

                speaker_influence = np.mean(speaker_alphas)

                # influence-weighted opinion
                speaker_position = np.sum((speaker_alphas / np.sum(speaker_alphas)) * speaker_opinions)
            else:
                speaker_ind = speaker_inds[0]
                speaker_influence = self.population[speaker_ind].alpha
                speaker_position = self.population[speaker_ind].opinion

            non_speaker_masks = np.ones(len(self.population), np.bool)
            non_speaker_masks[speaker_inds] = 0

            # select listeners
            listener_inds = np.random.choice(pop_inds[non_speaker_masks],
                                             size=self.lim_listeners, replace=False)

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

            self.n_event += 1
            # print(self.sd_opinion())

        # Calculate additional resource produced by the group (group-level payoff)
        prev_Bt = self.Bt
        self.Bt = (self.betab/(1 + np.exp(-self.gammab*(self.N - self.b_mid)))) -\
                  self.Ct * self.n_event
        self.Bt = max(0, self.Bt)

        # Calculate the share of resources each individual receives
        con_diff = np.abs(init_opinions - self.mean_opinion())
        alphar = 1 - con_diff

        pt = (0.001 + (self.d * alphar))
        pt = pt/np.sum(pt)

        # Calculate additional growth rate for each individual
        rb = self.betar * (1 - np.exp(-self.gammar * ((self.Bt + (self.S * prev_Bt)) * pt)))
        # rb[ rb < 0 ] = 0

        # Calculate fitness for each individual
        w = (self.ra / (1 + (self.N/self.K))) + rb
        w[ w < 0 ] = 0

        for i, agent in enumerate(self.population):
            agent.w = w[i]
            agent.con_dist = con_diff[i]


    def reproduce(self):
        np.random.seed()

        offsprings = []

        for agent in self.population:
            # might need to adjust opinion to inheritance mode
            o_n = np.random.poisson(agent.w)

            # randomize mutation
            mutation_masks = np.random.choice([0,1], p=[1-self.mu_rate, self.mu_rate], size=o_n)

            # adjusted truncated thresholds
            a, b = (0 - agent.alpha) / self.alpha_std, (1 - agent.alpha) / self.alpha_std

            # mutated alpha values
            o_alphas = truncnorm.rvs(a , b, loc=agent.alpha, scale=self.alpha_std, size=o_n)
            o_alphas = [agent.alpha if mutation_masks[i] == 0 else o_alphas[i] for i in range(o_n)]

            offspring_tmp = [OpinionAgent(alpha=o_alphas[i],
                                          opinion=np.random.rand(),
                                          cd=agent.con_dist,
                                          w=agent.w)
                             for i in range(o_n)]
            offsprings.extend(offspring_tmp)

        # reproduction step
        self.population = offsprings

        # update population parameter
        self.N = len(self.population)


    def die(self, survive_prob=1):

        cur_n = self.N
        new_n = int(np.round(cur_n * survive_prob))
        removed_n = cur_n - new_n

        for _ in range(removed_n):
            self.population.pop(np.random.randint(len(self.population)))

        self.N = len(self.population)


    def reset_opinion(self):
        for agent in self.population:
            agent.opinion = np.random.rand()


class EvoOpinionModel():
    """
    A consensus building model with opinion agents that have different levels of influence

    init_n: initial number of agents in each community
    x_threshold: consensus threshold for sd-criterion,
        but stands in for exponential decay parameter for proportional criterion
    k: exponent determining the mapping between influence and speaking probability
    lim_speakers: number of speakers per speaking event
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
    criterion: decision rule for the groups (either based on sd threshold or proportional threshold)
    init_cond: initial conditions for distribution of influence among agents
    load_communities: a set of communities provided to the model in a scenario
        where user wants to resume the simulation
    """
    def __init__(self, init_n, x_threshold, k, lim_speakers, lim_listeners, np, mu_rate, alpha_var,
                 K, ra, gammar, betar, gammab, betab, S, b_mid, Ct, d, m,
                 criterion='sd_threshold', init_cond='uniform', load_communities=None):
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
        self.d = d
        self.m = m
        self.criterion = criterion
        self.init_cond = init_cond
        self.step_count = 0
        self.communities = set() # or list()
        self.agent_reporter = {'group_id': lambda c: c.id,
                               'n_event': lambda c: c.n_event,
                               'group_size': lambda c: c.N,
                               # 'extra_resource': lambda c: c.Bt,
                               # 'avg_fitness': lambda c: c.mean_fitness(),
                               'mean_opinion': lambda c: c.mean_opinion(),
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
                                               N=self.init_n,
                                               x_threshold=self.x_threshold,
                                               k=self.k,
                                               lim_speakers=self.lim_speakers,
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
                                               d=self.d,
                                               criterion=self.criterion,
                                               init_cond=self.init_cond))

    # Model time step
    def step(self, capped_pop=False, verbose=False, process_num=1):

        self.form_decision(verbose=verbose, process_num=process_num)
        self.step_count += 1

        # collecting model results
        for key, collect_func in self.agent_reporter.items():
            self.datacollector[key].extend(list(map(collect_func, self.communities)))

        self.datacollector['step'].extend([self.step_count] * self.np)

        self.reproduce(process_num=process_num)
        self.migrate()

        if capped_pop:
            self.rescale_pop()

    def form_decision(self, verbose=False, process_num=1):

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

    def reproduce(self, process_num=1):

        # initiate multicore-processing pool
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

    def reset_opinion(self):
        for community in self.communities:
            community.reset_opinion()

    def migrate(self):

        # Migration step
        if self.np > 1 and self.m is not None:
            list_com = list(self.communities)
            c_migrants = []
            # prem_pop_dict = {}
            # groupm_dict = {}

            ## sample migrants from all communities
            for i in range(self.np):
                community = list_com[i]
                pop_ids = np.arange(len(community.population))
                # prem_pop_dict[i] = len(community.population) # group size for each community
                m_masks = np.random.binomial(size=pop_ids.shape[0], n=1, p=self.m)
                m_ids = pop_ids[m_masks == 1]

                # list of each group's migrants (list within list)
                c_migrants.append([community.population.pop(m_id) for m_id in reversed(m_ids)])
                # groupm_dict[i] = len(c_migrants[i])

            ## assign patches to all migrants
            c_ids = np.arange(self.np)
            # mto_list = []
            for i in range(self.np):
                new_c_ids = np.random.choice(c_ids[c_ids != i], size=len(c_migrants[i]), replace=True)
                # mto_list.extend(new_c_ids)

                for c_id in new_c_ids:
                    list_com[c_id].population.append(c_migrants[i].pop(0))

            # mto_dict = dict(sorted(Counter(mto_list).items()))

            ## update group size
            # postm_pop_dict = {}
            for i, c in enumerate(self.communities):
                c.N = len(c.population)
                # postm_pop_dict[i] = c.N

            # print('Pre-migration size = ', prem_pop_dict)
            # print('Migrants from each group = ', groupm_dict)
            # print('Migrants to each group = ', mto_dict)
            # print('Post-migration size = ', postm_pop_dict)

    # rescaling total population to a fixed size
    def rescale_pop(self, process_num=1):

        fixed_n = self.init_n * self.np
        total_n = 0

        for community in self.communities:
            total_n += community.N

        print('Pre-rescaling population size = ', total_n)

        if fixed_n < total_n:

            rescaled_coef = fixed_n / total_n
            print(rescaled_coef)

            # initiate multicore-processing pool
            if process_num > 1:
                pool = mp.Pool(processes=process_num)

            # Reproduction step
            if process_num == 1:
                for community in self.communities:
                    community.die(survive_prob=rescaled_coef)
            else:
                for community in self.communities:
                    pool.apply_async(community.die(rescaled_coef))

                pool.close()
                pool.join()

            tmp_n = 0
            for community in self.communities:
                tmp_n += community.N

            print('Post-rescaling pop = ', tmp_n)


    def save_communities(self, filepath):
        # with open(filepath, 'wb') as file:
        joblib.dump(self.communities, filepath)