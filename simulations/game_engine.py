import numpy as np
import networkx as nx
from scipy.stats import norm
from scipy.special import log_ndtr


class Agent:
    """
    creates an agent with index n, private llr a, social llr b
    handles:
    decision making based on private and social llrs passed to it
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

    handles:
    signal generation
    game playing and tracking
    metrics for convergence and running accuracy
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
        self.belief_engine = BeliefEngine(
            self.N, graph_type, signal_type, self.q, graph
        )

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

    def play(self):
        """
        plays sequential game depending on graph and signal type
        """
        sorted_nodes = sorted(list(self.graph.nodes))
        history_array = np.zeros(self.N, dtype=int)
        for n in sorted_nodes:
            sn = self.draw_signal()
            an = self.belief_engine.priv_llr(sn)
            bn = self.belief_engine.social_llr(n, history_array)
            agent = Agent(n, an, bn, rng=self.rng)
            action = agent.decide()
            self.agents[n] = agent
            self.history[n] = action
            history_array[n] = action
            self.belief_engine.update_beliefs(an, action)

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


class BeliefEngine:
    """
    creates a belief engine to compute posterior beliefs based on actions
    handles:
    computing private and social log-likelihood ratios
    updating beliefs based on actions
    """
    def __init__(self, N, graph_type, signal_type, q, graph):
        self.N = N
        self.graph_type = graph_type
        self.signal_type = signal_type
        self.q = q

        self.adj_matrix = \
            nx.to_scipy_sparse_array(graph, format='csr').T.tocsr()

        self.bn = 0.0
        self.alpha = self.q if self.signal_type == "bounded" else norm.cdf(1)
        self.beta = 1 - self.alpha
        if graph_type == "previous":
            self.bn0 = np.log(self.alpha / self.beta)
        else:
            self.bn0 = 0.0
        if graph_type == "previous":
            self.bn1 = np.log((1 - self.alpha) / (1 - self.beta))
        else:
            self.bn1 = 0.0

    def priv_llr(self, s):
        """
        returns log-likelihood ratio for agent's private signal
        """
        if self.signal_type == "bounded":
            if s:
                return np.log((1 - self.q) / self.q)
            return np.log(self.q / (1 - self.q))
        return 2 * s

    def social_llr(self, n, history_array):
        """
        computes social log-likelihood ratio for agent n based on history
        """
        if self.graph_type == "complete":
            return self.bn

        elif self.graph_type == "previous":
            if n == 0:
                return 0.0
            return self.bn1 if history_array[n - 1] else self.bn0

        else:
            ptr_start = self.adj_matrix.indptr[n]
            ptr_end = self.adj_matrix.indptr[n+1]
            if ptr_start == ptr_end:
                return 0.0

            nbd = self.adj_matrix.indices[ptr_start:ptr_end]
            obs = history_array[nbd]
            num_ones = np.sum(obs)
            num_zeros = len(obs) - num_ones

            return (num_ones * np.log((1 - self.q) / self.q)) + \
                   (num_zeros * np.log(self.q / (1 - self.q)))

    def update_beliefs(self, an, action):
        """
        updates social log-likelihood ratio for next agent
        """
        if self.graph_type == "complete":
            if self.signal_type == "bounded":
                wn = abs(an)

                if np.isclose(self.bn, wn, atol=1e-8):
                    if action:
                        self.bn += np.log((1 - self.q) / self.q)
                    else:
                        self.bn += np.log((1 + self.q) / (2 - self.q))

                elif np.isclose(self.bn, -wn, atol=1e-8):
                    if action:
                        self.bn += np.log((2 - self.q) / (1 + self.q))
                    else:
                        self.bn += np.log(self.q / (1 - self.q))

                elif -wn < self.bn < wn:
                    if action:
                        self.bn += np.log((1 - self.q) / self.q)
                    else:
                        self.bn += np.log(self.q / (1 - self.q))

                else:
                    pass  # bn remains unchanged if outside [-wn, wn]

            else:
                if action:
                    self.bn += log_ndtr(-(self.bn/2) - 1)\
                        - log_ndtr(-(self.bn/2) + 1)
                else:
                    self.bn += log_ndtr((self.bn/2) + 1)\
                        - log_ndtr((self.bn/2) - 1)

        elif self.graph_type == "previous":
            if self.signal_type == "bounded":
                wn = abs(an)
                old_alpha = self.alpha
                old_beta = self.beta

                # first alpha beta update
                if np.isclose(self.bn0, wn, atol=1e-8):
                    self.alpha = (1 + self.q) * old_alpha / 2
                    self.beta = (2 - self.q) * old_beta / 2
                elif np.isclose(self.bn0, -wn, atol=1e-8):
                    self.alpha = self.q * old_alpha / 2
                    self.beta = (1 - self.q) * old_beta / 2
                elif -wn < self.bn0 < wn:
                    self.alpha = self.q * old_alpha
                    self.beta = (1 - self.q) * old_beta
                elif self.bn0 > wn:
                    self.alpha = old_alpha
                    self.beta = old_beta
                elif self.bn0 < -wn:
                    self.alpha = self.beta = 0

                # second alpha beta update
                if np.isclose(self.bn1, wn, atol=1e-8):
                    self.alpha += (1 + self.q) * (1 - old_alpha) / 2
                    self.beta += (2 - self.q) * (1 - old_beta) / 2
                elif np.isclose(self.bn1, -wn, atol=1e-8):
                    self.alpha += self.q * (1 - old_alpha) / 2
                    self.beta += (1 - self.q) * (1 - old_beta) / 2
                elif -wn < self.bn1 < wn:
                    self.alpha += self.q * (1 - old_alpha)
                    self.beta += (1 - self.q) * (1 - old_beta)
                elif self.bn1 > wn:
                    self.alpha += 1 - old_alpha
                    self.beta += 1 - old_beta
                elif self.bn1 < -wn:
                    pass

            else:
                self.alpha = self.alpha * norm.cdf((self.bn0/2) + 1)\
                    + (1 - self.alpha) * norm.cdf((self.bn1/2) + 1)
                self.beta = self.beta * norm.cdf((self.bn0/2) - 1)\
                    + (1 - self.beta) * norm.cdf((self.bn1/2) - 1)

            eps = 1e-15
            self.alpha = np.clip(self.alpha, eps, 1 - eps)
            self.beta = np.clip(self.beta, eps, 1 - eps)

            self.bn0 = np.log(self.alpha / self.beta)
            self.bn1 = np.log((1 - self.alpha) / (1 - self.beta))
