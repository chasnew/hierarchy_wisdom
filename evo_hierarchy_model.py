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
    def __init__(self, alpha, opinion, cid, w=0, cd=0, mv=0):
        self.alpha = alpha # influence
        self.opinion = opinion
        self.cid = cid
        self.w = w
        self.con_dist = cd
        self.mv_dist = mv

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
        self.init_cond = init_cond

        self.n_event = 0
        self.Bt = 0 # additional resource produced by group

        # create a population of agents
        self.population = self.create_population(self.init_cond)

    def create_population(self, init_cond='uniform'):
        population = []

        if init_cond == 'uniform':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.rand(),
                                               opinion=np.random.rand(),
                                               cid=self.id))
        elif init_cond == 'most_leaders':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.beta(a=1.9,b=0.1,size=1)[0],
                                               opinion=np.random.rand(),
                                               cid=self.id))
        elif init_cond == 'most_followers':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.beta(a=0.1,b=1.9,size=1)[0],
                                               opinion=np.random.rand(),
                                               cid=self.id))
        elif init_cond == 'left_skew':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.beta(a=1.5,b=0.5,size=1)[0],
                                               opinion=np.random.rand(),
                                               cid=self.id))
        elif init_cond == 'right_skew':
            for i in range(self.N):
                population.append(OpinionAgent(alpha=np.random.beta(a=0.5,b=1.5,size=1)[0],
                                               opinion=np.random.rand(),
                                               cid=self.id))

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
        # self.Bt = (self.betab/(1 + np.exp(-self.gammab*(self.N - self.b_mid)))) -\
        #           self.Ct * self.n_event
        self.Bt = self.betab - self.Ct * self.n_event # no group size effect
        self.Bt = max(0, self.Bt)

        # Calculate the share of resources each individual receives
        con_diff = np.abs(init_opinions - self.mean_opinion())
        alphar = 1 - con_diff

        pt = (1 + (self.d * alphar))
        pt = pt/np.sum(pt)

        # Calculate additional growth rate for each individual
        rb = self.betar * (1 - np.exp(-self.gammar * ((self.Bt + (self.S * prev_Bt)) * pt)))
        rb[ rb < 0 ] = 0

        # Calculate fitness for each individual
        # w = (self.ra / (1 + (self.N/self.K))) + rb
        w = self.ra + rb

        for i, agent in enumerate(self.population):
            agent.w = w[i]
            agent.con_dist = con_diff[i]
            agent.mv_dist = np.abs(agent.opinion - init_opinions[i])


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
                                          cid=self.id,
                                          cd=agent.con_dist,
                                          w=agent.w)
                             for i in range(o_n)]
            offsprings.extend(offspring_tmp)

        # reproduction step
        self.population = offsprings

        # update population parameter
        self.N = len(self.population)


    def die(self, death_rate=0):

        cur_n = self.N
        removed_n = int(np.round(cur_n * death_rate))

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
    ng: number of groups
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
    dr: death rate
    criterion: decision rule for the groups (either based on sd threshold or proportional threshold)
    init_cond: initial conditions for distribution of influence among agents
    load_communities: a set of communities provided to the model in a scenario
        where user wants to resume the simulation
    """
    def __init__(self, init_n, x_threshold, k, lim_speakers, lim_listeners, ng, mu_rate, alpha_var,
                 K, ra, gammar, betar, gammab, betab, S, b_mid, Ct, d, m, dr=0,
                 criterion='sd_threshold', init_cond='uniform', load_communities=None):

        # rescale base payoff for cases when slow decision is favored
        if Ct < 0:
            betab = 0

        self.init_n = init_n
        self.x_threshold = x_threshold
        self.k = k
        self.lim_speakers = lim_speakers
        self.lim_listeners = lim_listeners
        self.ng = ng
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
        self.dr = dr
        self.criterion = criterion
        self.init_cond = init_cond
        self.step_count = 0
        self.communities = list()
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
            init_cond_set = ['uniform', 'most_leaders', 'most_followers', 'left_skew', 'right_skew']
            if self.init_cond == 'randomized':
                init_conds = np.random.choice(init_cond_set, self.ng)
            elif self.init_cond == 'even':
                ng_per_cond = int(np.ceil(self.ng / 5))
                res = self.ng % 5
                init_conds = []
                if res == 0:
                    for i in range(5):
                        init_conds.extend([init_cond_set[i]] * ng_per_cond)
                else:
                    for i in range(res):
                        init_conds.extend([init_cond_set[i]] * ng_per_cond)
                    for i in range(res, 5):
                        init_conds.extend([init_cond_set[i]] * (ng_per_cond - 1))
            elif self.init_cond in init_cond_set:
                init_conds = [self.init_cond] * self.ng
            else:
                init_conds = []
                cond_num = self.init_cond.split(',')
                for i in range(5):
                    init_conds.extend([init_cond_set[i]] * int(cond_num[i]))

            print(init_conds)

            for i in range(self.ng):
                self.communities.append(Community(unique_id=i,
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
                                                  init_cond=init_conds[i]))

    # Model time step
    def step(self, fixed_pop=False, verbose=False, process_num=1):

        self.form_decision(verbose=verbose, process_num=process_num)
        self.step_count += 1

        # collecting model results
        for key, collect_func in self.agent_reporter.items():
            self.datacollector[key].extend(list(map(collect_func, self.communities)))

        self.datacollector['step'].extend([self.step_count] * self.ng)

        if fixed_pop:
            self.rescale_pop(process_num=process_num)
        else:
            self.reproduce(process_num=process_num)

        self.migrate()

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

    def migrate(self, fixed_popsize=False):

        # Migration step
        if self.ng > 1 and self.m is not None:
            c_migrants = []
            # prem_pop_dict = {}
            # groupm_dict = {}

            ## migrate agents so that the size of all groups stay constant
            if fixed_popsize == 'all_groups':

                ## sample migrants from all communities
                for i in range(self.ng):
                    community = self.communities[i]
                    cur_n = len(community.population)

                    if cur_n > self.init_n:
                        num_migrant = (cur_n - self.init_n) + int(self.init_n * self.m)
                    else:
                        num_migrant = int(cur_n * self.m)

                    pop_ids = np.arange(len(community.population))
                    m_ids = np.random.choice(pop_ids, size=num_migrant, replace=False)
                    m_ids[::-1].sort() # sort id in descending order

                    # list of each group's migrants (list within list)
                    c_migrants.extend([community.population.pop(m_id) for m_id in m_ids])

                    community.N = len(community.population)

                ## assign communities to all migrants
                c_migrants = np.array(c_migrants)
                for i in range(self.ng):
                    community = self.communities[i]
                    new_memb_num = self.init_n - community.N

                    m_ids = np.arange(len(c_migrants))

                    # fill up community i with new members from random migrants
                    # print('migrant index length =', len(m_ids))
                    new_memb_ids = np.random.choice(m_ids, size=new_memb_num, replace=False)
                    community.population.extend(c_migrants[new_memb_ids])

                    # assign new community id
                    for m_id in new_memb_ids:
                        c_migrants[m_id].cid = i

                    # remove already-assigned migrants
                    mask = np.ones(len(m_ids), dtype=bool)
                    mask[new_memb_ids] = False
                    c_migrants = c_migrants[mask]

            else:
                ## sample migrants from all communities
                for i in range(self.ng):
                    community = self.communities[i]
                    pop_ids = np.arange(len(community.population))
                    m_masks = np.random.binomial(size=pop_ids.shape[0], n=1, p=self.m)
                    m_ids = pop_ids[m_masks == 1]

                    # list of each group's migrants (list within list)
                    c_migrants.append([community.population.pop(m_id) for m_id in reversed(m_ids)])

                ## assign communities to all migrants
                c_ids = np.arange(self.ng)
                for i in range(self.ng):
                    # randomize a new community for migrants from community i
                    new_c_ids = np.random.choice(c_ids[c_ids != i], size=len(c_migrants[i]), replace=True)

                    for c_id in new_c_ids:
                        c_migrant = c_migrants[i].pop(0)
                        c_migrant.cid = c_id
                        self.communities[c_id].population.append(c_migrant)


            ## update group size
            for i, c in enumerate(self.communities):
                c.N = len(c.population)


    # rescaling total population to a fixed size
    def rescale_pop(self, process_num=1):

        fixed_n = self.init_n * self.ng

        # initiate multicore-processing pool
        if process_num > 1:
            pool = mp.Pool(processes=process_num)

        # Reproduction step
        if process_num == 1:
            for community in self.communities:
                community.die(death_rate=self.dr)
        else:
            for community in self.communities:
                pool.apply_async(community.die(self.dr))

            pool.close()
            pool.join()

        meta_pop = []

        for community in self.communities:
            meta_pop += community.population

        cur_n = len(meta_pop)
        birth_n = fixed_n - cur_n

        meta_w = []
        for agent in meta_pop:
            meta_w.append(agent.w)

        reproduce_prob = meta_w / np.sum(meta_w)

        # sample parents in proportion to their fitness
        parents = np.random.choice(meta_pop, size=birth_n, replace=True, p=reproduce_prob)

        # Reproduction proportional to fitness
        for par_id, parent in enumerate(parents):
            cid = parent.cid

            offspring_tmp = OpinionAgent(alpha=parent.alpha,
                                         opinion=parent.opinion,
                                         cid=cid,
                                         cd=parent.con_dist,
                                         w=parent.w)

            self.communities[cid].population.append(offspring_tmp)

        # reset opinions and update group size
        new_meta_pop = []

        for community in self.communities:
            new_meta_pop += community.population
            community.N = len(community.population)
            community.reset_opinion()

        # alpha mutation
        mutation_masks = np.random.choice([0, 1], p=[1 - self.mu_rate, self.mu_rate], size=fixed_n)

        alpha_std = np.sqrt(self.alpha_var)

        for _id, agent in enumerate(new_meta_pop):
            if mutation_masks[_id]:
                # adjusted truncated thresholds
                a, b = (0 - agent.alpha) / alpha_std, (1 - agent.alpha) / alpha_std
                agent.alpha = truncnorm.rvs(a, b, loc=agent.alpha, scale=alpha_std, size=1)[0]


    def save_communities(self, filepath):
        # with open(filepath, 'wb') as file:
        joblib.dump(self.communities, filepath)