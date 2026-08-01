import numpy as np
from scipy.stats import norm

# class Agent:
#     def __init__(self, n, s, q, llrf, sllr=0, rng=None):
#         """
#         i: index of agent
#         s: private signal of agent
#         q: accuracy of neighbours' actions
#         llrf: function to calculate log-likelihood ratio from private signal
#         rng: random number generator for tie-breaking
#         """
#         self.n = n  # index
#         self.s = s  # private signal
#         self.q = q  # neighbour accuracy (= signal accuracy)
#         self.sllr = sllr  # social log-likelihood ratio
#         self.llrf = llrf  # function used to calculate llr
#         self.rng = np.random.default_rng() if rng is None else rng
#         self.action = None

#     def naive_decide(self, num_ones, num_zeros):
#         """
#         contains naive Bayesian updating
#         note: this is perfect Bayesian updating for NEO graph used
#         """
#         priv_llr = self.llrf(self.s)

#         llr_0 = np.log(self.q / (1 - self.q))
#         llr_1 = -llr_0

#         soc_llr = (num_ones * llr_1) + (num_zeros * llr_0)

#         llr = priv_llr + soc_llr

#         if np.isclose(llr, 0, atol=1e-8):
#             self.action = self.rng.choice([0, 1])
#         elif llr > 0:
#             self.action = 0
#         elif llr < 0:
#             self.action = 1

#         return self.action

#     def llr(self):
#         """
#         returns log-likelihood ratio for agent's private signal
#         """
#         return self.llrf(self.s) + self.sllr

#     def det_decide(self):
#         """
#         contains perfect Bayesian updating for deterministic graph
#         """
#         llr = self.llr()

#         if np.isclose(llr, 0, atol=1e-8):
#             self.action = self.rng.choice([0, 1])
#         elif llr > 0:
#             self.action = 0
#         elif llr < 0:
#             self.action = 1

#         return self.action

#     # def new_sllr(self):
#     #     """
#     #     returns social log-likelihood ratio for next agent
#     #     """
#     #     return self.llrf(self.s) + 2 * self.sllr


# class SequentialGame:
#     def __init__(self, graph, signal_type, q=None, rng=None):
#         """
#         graph: directed graph chosen based on a specified network topology
#         signal_type: 'bounded' or 'unbounded'
#         q: signal accuracy (only relevant for bounded signals)
#         rng: random number generator for reproducibility
#         """
#         self.graph = graph
#         self.N = len(graph.nodes)  # number of agents
#         self.signal_type = signal_type  # bounded or unbounded
#         self.rng = np.random.default_rng() if rng is None else rng
#         self.true_state = self.rng.choice([0, 1])  # theta

#         if signal_type == "bounded":  # signal accuracy
#             self.q = q
#         elif signal_type == "unbounded":
#             self.q = 0.8413
#         else:
#             raise ValueError("Invalid signal_type provided.")

#         self.agents = {}
#         self.history = {}

#     def draw_signal(self):
#         """
#         draws private signal for agent based on true state and signal type
#         """
#         if self.signal_type == "bounded":
#             if self.rng.random() < self.q:
#                 return self.true_state
#             return 1 - self.true_state

#         if self.true_state:
#             return self.rng.normal(-1, 1)
#         return self.rng.normal(1, 1)

#     def get_llrf(self):
#         """
#         returns function to calculate log-likelihood ratio
#         depends on signal type
#         """
#         if self.signal_type == "bounded":
#             return lambda s: np.log((1 - self.q) / self.q) if s\
#                 else np.log(self.q / (1 - self.q))
#         return lambda s: 2 * s

#     def play(self):
#         """
#         plays sequential game for all agents
#         """
#         sorted_nodes = sorted(list(self.graph.nodes))
#         llrf = self.get_llrf()

#         history_array = np.zeros(self.N, dtype=int)
#         ratio_array = np.zeros(self.N, dtype=float)

#         sllr = 0  # initial social log-likelihood ratio
#         for n in sorted_nodes:
#             sn = self.draw_signal()
#             agent = Agent(n, sn, self.q, llrf, sllr=sllr, rng=self.rng)
#             nbd = list(self.graph.successors(n))

#             # if nbd:
#             #     sllr = ratio_array[nbd].sum()

#             # agent = Agent(n, sn, self.q, llrf, sllr=sllr, rng=self.rng)
#             # action = agent.det_decide()
#             if nbd:
#                 obs = history_array[nbd]  # pull all neighbour actions
#                 num_ones = np.sum(obs)  # count 1s
#                 num_zeros = len(obs) - num_ones  # rest are 0s
#             else:
#                 num_ones = 0
#                 num_zeros = 0

#             action = agent.naive_decide(num_ones, num_zeros)

#             self.agents[n] = agent
#             self.history[n] = action
#             history_array[n] = action
#             ratio_array[n] = agent.llr()  # store log-likelihood ratio

#     def convergence_metrics(self, threshold=None):
#         """
#         returns whether game converged, whether converged to true state,
#         index of first agent in lock-in streak, final accuracy
#         """
#         if not self.history:
#             return False, None, None, None

#         if threshold is None:
#             threshold = self.N // 5

#         final_action = list(self.history.values())[-1]
#         actions = list(self.history.values())

#         lock_in_index = 0
#         for t in range(self.N - 1, -1, -1):
#             if actions[t] != final_action:
#                 lock_in_index = t + 1
#                 break

#         streak_length = self.N - lock_in_index

#         if streak_length >= threshold:
#             success = bool(final_action == self.true_state)
#             return True, success, lock_in_index, self.running_accuracy()[-1]
#         return False, None, None, self.running_accuracy()[-1]

#     def running_accuracy(self):
#         """
#         returns array of running accuracy of actions compared to true state
#         """
#         actions = np.array(list(self.history.values()))
#         correct_guesses = actions == self.true_state
#         return np.cumsum(correct_guesses) / np.arange(1, self.N + 1)


class Agent:
    """
    creates an agent with index n, private llr a, social llr b
    """
    def __init__(self, n, an, bn, rng=None):
        self.n = n
        self.an = an
        self.bn = bn
        self.rng = np.random.default_rng() if rng is None else rng

    def decide(self):
        """
        returns action based on private and social log-likelihood ratios
        """
        llr = self.an + self.bn
        if np.isclose(llr, 0, atol=1e-8):
            return self.rng.choice([0, 1])
        elif llr > 0:
            return 0
        return 1


class SequentialGame:
    """
    creates a sequential game with:
    graph = directed graph on which to play the game
    graph_type
    signal_type = 'bounded' or 'unbounded
    q = signal accuracy
    rng = random number generator for reproducibility
    """
    def __init__(self, graph, graph_type, signal_type, q=None, rng=None):
        self.graph = graph
        self.N = len(graph.nodes)
        self.graph_type = graph_type
        self.signal_type = signal_type
        self.q = q if signal_type == "bounded" else norm.cdf(1)
        self.rng = np.random.default_rng() if rng is None else rng
        self.true_state = self.rng.choice([0, 1])
        self.agents = {}
        self.history = {}

    def draw_signal(self):
        """
        draws private signal for agent based on true state and signal type
        """
        if self.signal_type == "bounded":
            if self.rng.random() < self.q:
                return self.true_state
            return 1 - self.true_state

        if self.true_state:
            return self.rng.normal(-1, 1)
        return self.rng.normal(1, 1)

    def pllr(self, s):
        """
        returns log-likelihood ratio for agent's private signal
        """
        if self.signal_type == "bounded":
            if s:
                return np.log((1 - self.q) / self.q)
            return np.log(self.q / (1 - self.q))
        return 2 * s

    def play(self):
        """
        plays sequential game depending on graph and signal type
        """
        sorted_nodes = sorted(list(self.graph.nodes))
        bn = 0  # initial social log-likelihood ratio
        if self.graph_type == "complete":
            for n in sorted_nodes:
                sn = self.draw_signal()
                an = self.pllr(sn)
                agent = Agent(n, an, bn, rng=self.rng)
                action = agent.decide()
                self.agents[n] = agent
                self.history[n] = action
                if self.signal_type == "bounded":
                    wn = abs(an)

                    if np.isclose(bn, wn, atol=1e-8):
                        if action:
                            bn += np.log((1 - self.q) / self.q)
                        else:
                            bn += np.log((1 + self.q) / (2 - self.q))

                    elif np.isclose(bn, -wn, atol=1e-8):
                        if action:
                            bn += np.log((2 - self.q) / (1 + self.q))
                        else:
                            bn += np.log(self.q / (1 - self.q))

                    elif -wn < bn < wn:
                        if action:
                            bn += np.log((1 - self.q) / self.q)
                        else:
                            bn += np.log(self.q / (1 - self.q))

                    else:
                        pass  # bn remains unchanged if outside [-wn, wn]

                else:
                    if action:
                        bn += np.log((norm.cdf(-(bn/2) - 1)
                                      / norm.cdf(-(bn/2) + 1)))
                    else:
                        bn += np.log((norm.cdf((bn/2) + 1)
                                      / norm.cdf((bn/2) - 1)))

        elif self.graph_type == "previous":
            sn = self.draw_signal()
            an = self.pllr(sn)
            agent = Agent(sorted_nodes[0], an, bn, rng=self.rng)
            action = agent.decide()
            self.agents[sorted_nodes[0]] = agent
            self.history[sorted_nodes[0]] = action

            if self.signal_type == "bounded":
                alpha = self.q
                beta = 1 - self.q
            else:
                alpha = norm.cdf(1)
                beta = 1 - alpha
            for n in sorted_nodes[1:]:
                sn = self.draw_signal()
                an = self.pllr(sn)
                bn0 = np.log(alpha / beta)
                bn1 = np.log((1 - alpha) / (1 - beta))
                if self.history[n - 1]:
                    agent = Agent(n, an, bn1, rng=self.rng)
                else:
                    agent = Agent(n, an, bn0, rng=self.rng)
                action = agent.decide()
                self.agents[n] = agent
                self.history[n] = action
                if self.signal_type == "bounded":
                    wn = abs(an)
                    old_alpha = alpha
                    old_beta = beta

                    # first alpha beta update
                    if np.isclose(bn0, wn, atol=1e-8):
                        alpha = (1 + self.q) * old_alpha / 2
                        beta = (2 - self.q) * old_beta / 2
                    elif np.isclose(bn0, -wn, atol=1e-8):
                        alpha = self.q * old_alpha / 2
                        beta = (1 - self.q) * old_beta / 2
                    elif -wn < bn0 < wn:
                        alpha = self.q * old_alpha
                        beta = (1 - self.q) * old_beta
                    elif bn0 > wn:
                        alpha = old_alpha
                        beta = old_beta
                    elif bn0 < -wn:
                        alpha = beta = 0

                    # second alpha beta update
                    if np.isclose(bn1, wn, atol=1e-8):
                        alpha += (1 + self.q) * (1 - old_alpha) / 2
                        beta += (2 - self.q) * (1 - old_beta) / 2
                    elif np.isclose(bn1, -wn, atol=1e-8):
                        alpha += self.q * (1 - old_alpha) / 2
                        beta += (1 - self.q) * (1 - old_beta) / 2
                    elif -wn < bn1 < wn:
                        alpha += self.q * (1 - old_alpha)
                        beta += (1 - self.q) * (1 - old_beta)
                    elif bn1 > wn:
                        alpha += 1 - old_alpha
                        beta += 1 - old_beta
                    elif bn1 < -wn:
                        pass

                else:
                    alpha = alpha * norm.cdf((bn0/2) + 1)\
                        + (1 - alpha) * norm.cdf((bn1/2) + 1)
                    beta = beta * norm.cdf((bn0/2) - 1)\
                        + (1 - beta) * norm.cdf((bn1/2) - 1)
        else:
            history_array = np.zeros(self.N, dtype=int)
            for n in sorted_nodes:
                sn = self.draw_signal()
                an = self.pllr(sn)
                nbd = list(self.graph.successors(n))
                if nbd:
                    obs = history_array[nbd]
                    num_ones = np.sum(obs)
                    num_zeros = len(obs) - num_ones
                    bn = (num_ones * np.log((1 - self.q) / self.q))\
                        + (num_zeros * np.log(self.q / (1 - self.q)))
                else:
                    bn = 0
                agent = Agent(n, an, bn, rng=self.rng)
                action = agent.decide()
                self.agents[n] = agent
                self.history[n] = action
                history_array[n] = action

    def convergence_metrics(self, threshold=None):
        """
        returns whether game converged, whether converged to true state,
        index of first agent in lock-in streak, final accuracy
        """
        if not self.history:
            return False, None, None, None

        if threshold is None:
            threshold = self.N // 5

        final_action = list(self.history.values())[-1]
        actions = list(self.history.values())

        lock_in_index = 0
        for t in range(self.N - 1, -1, -1):
            if actions[t] != final_action:
                lock_in_index = t + 1
                break

        streak_length = self.N - lock_in_index

        if streak_length >= threshold:
            success = bool(final_action == self.true_state)
            return True, success, lock_in_index, self.running_accuracy()[-1]
        return False, None, None, self.running_accuracy()[-1]

    def running_accuracy(self):
        """
        returns array of running accuracy of actions compared to true state
        """
        actions = np.array(list(self.history.values()))
        correct_guesses = actions == self.true_state
        return np.cumsum(correct_guesses) / np.arange(1, self.N + 1)
