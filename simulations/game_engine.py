import numpy as np
import networkx as nx
from scipy.stats import norm
from scipy.special import log_ndtr


class SequentialGame:
    """
    creates a sequential game with:
    graph = directed graph on which to play the game
    graph_type (str)
    signal_type = 'bounded' or 'unbounded
    q = signal accuracy
    rng = random number generator for reproducibility
    k = number of royals
    p = probability of edge in ER graph
    sample = number of neighbours to sample in BS graph
    M = number of Monte Carlo simulations for ER and BS graphs

    handles:
    signal generation
    decision making
    game playing and tracking
    metrics for convergence and running accuracy
    """
    def __init__(self,
                 graph,
                 graph_type,
                 signal_type,
                 rng=None,
                 q=None,
                 k=None,
                 p=None,
                 sample=None,
                 M=None
                 ):
        self.graph = graph
        self.graph_type = graph_type
        self.signal_type = signal_type
        self.N = len(graph.nodes)
        self.rng = np.random.default_rng() if rng is None else rng
        self.q = q if signal_type == "bounded" else norm.cdf(1)
        self.k = k
        self.p = p
        self.sample = sample
        self.M = M

        self.true_state = self.rng.choice([0, 1])
        self.history = np.zeros(self.N, dtype=int)
        self.played = False

        self.belief_engine = BeliefEngine(
            graph=self.graph,
            graph_type=self.graph_type,
            signal_type=self.signal_type,
            N=self.N,
            rng=self.rng,
            q=self.q,
            k=self.k,
            p=self.p,
            sample=self.sample,
            M=self.M,
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

    def decide(self, an, bn):
        llr = an + bn
        if np.isclose(llr, 0, atol=1e-8):
            action = self.rng.choice([0, 1])
        elif llr > 0:
            action = 0
        else:
            action = 1
        return action

    def play(self):
        """
        plays sequential game depending on graph and signal type
        """
        sorted_nodes = sorted(list(self.graph.nodes))
        for n in sorted_nodes:
            sn = self.draw_signal()
            an = self.belief_engine.priv_llr(sn)
            bn = self.belief_engine.social_llr(n, self.history)
            action = self.decide(an, bn)
            self.history[n] = action
            self.belief_engine.update_beliefs(an, action)
        self.played = True

    def convergence_metrics(self, threshold=None):
        """
        returns whether game converged, whether converged to true state,
        index of first agent in lock-in streak, final accuracy
        """
        if not self.played:
            return False, None, None, None

        if threshold is None:
            threshold = self.N // 5

        final_action = self.history[-1]

        mismatches = np.where(self.history != final_action)[0]
        lock_in_index = mismatches[-1] + 1 if len(mismatches) > 0 else 0
        streak_length = self.N - lock_in_index

        if streak_length >= threshold:
            success = bool(final_action == self.true_state)
            return True, success, lock_in_index, self.running_accuracy()[-1]
        return False, None, None, self.running_accuracy()[-1]

    def running_accuracy(self):
        """
        returns array of running accuracy of actions compared to true state
        """
        correct_guesses = self.history == self.true_state
        return np.cumsum(correct_guesses) / np.arange(1, self.N + 1)


class BeliefEngine:
    """
    creates a belief engine to compute posterior beliefs based on actions
    handles:
    computing private and social log-likelihood ratios
    updating beliefs based on actions
    """
    def __init__(self,
                 graph,
                 graph_type,
                 signal_type,
                 N,
                 rng=None,
                 q=None,
                 k=None,
                 p=None,
                 sample=None,
                 M=None
                 ):
        self.adj_matrix = \
            nx.to_scipy_sparse_array(graph, format='csr').tocsr()
        self.graph_type = graph_type
        self.signal_type = signal_type
        self.N = N
        # rng for Monte Carlo simulations in ER and BS graphs
        self.rng = np.random.default_rng() if rng is None else rng

        self.q = q
        self.Q = max(q, 1-q)
        # exact belief update parameter initialisations
        self.bn = 0.0
        if graph_type == "previous":
            self.alpha = (self.Q if self.signal_type == "bounded"
                          else norm.cdf(1))
            self.beta = 1 - self.alpha
            self.bn0 = np.log(self.alpha / self.beta)
            self.bn1 = np.log((1 - self.alpha) / (1 - self.beta))
        else:
            self.bn0 = 0
            self.bn1 = 0

        if graph_type == "NEO":
            self.k = k
        elif graph_type == "ER":
            self.p = p
        elif graph_type == "BS":
            self.sample = sample

        if self.graph_type in ["ER", "BS"]:
            self.M = int(M) if M is not None else 10000
            self.M += self.M % 2
            self.halfM = self.M // 2
            if self.signal_type == "bounded":
                draws0 = self.rng.random((self.halfM, self.N)) < self.q
                draws1 = self.rng.random((self.halfM, self.N)) < self.q
                signal_matrix = np.vstack((1 - draws0, draws1)).astype(int)
                llr0 = np.log(self.q / (1 - self.q))
                llr1 = np.log((1 - self.q) / self.q)
                self.M_priv_llr = np.where(signal_matrix == 0, llr0, llr1)
            else:
                draws0 = self.rng.normal(1, 1, size=(self.halfM, self.N))
                draws1 = self.rng.normal(-1, 1, size=(self.halfM, self.N))
                signal_matrix = np.vstack((draws0, draws1))
                self.M_priv_llr = 2 * signal_matrix
            self.M_actions = np.zeros((self.M, self.N), dtype=int)
            self.M_running_ones = np.zeros(self.M, dtype=int)
            self.mc_computed_upto = 0
            self._precompute_mc_actions()

    def priv_llr(self, s):
        """
        returns log-likelihood ratio for agent's private signal
        """
        if self.signal_type == "bounded":
            if s:
                return np.log((1 - self.q) / self.q)
            return np.log(self.q / (1 - self.q))
        return 2 * s

    def social_llr(self, n, history):
        """
        computes social log-likelihood ratio for agent n based on history
        """
        if self.graph_type == "complete":
            return self.bn

        elif self.graph_type == "previous":
            if n == 0:
                return 0
            return self.bn1 if history[n - 1] else self.bn0

        else:
            ptr_start = self.adj_matrix.indptr[n]
            ptr_end = self.adj_matrix.indptr[n+1]
            if ptr_start == ptr_end:
                return 0

            nbd = self.adj_matrix.indices[ptr_start:ptr_end]
            obs = history[nbd]
            if self.graph_type == "NEO":
                num_ones = np.sum(obs)
                num_zeros = len(obs) - num_ones

                return (num_ones * np.log((1 - self.Q) / self.Q)) + \
                    (num_zeros * np.log(self.Q / (1 - self.Q)))

            return self._mc_soc_llr(n, nbd, obs, history)

    def update_beliefs(self, an, action):
        """
        updates social log-likelihood ratio for next agent
        """
        if self.graph_type == "complete":
            if self.signal_type == "bounded":
                wn = abs(an)

                if np.isclose(self.bn, wn, atol=1e-8):
                    if action:
                        self.bn += np.log((1 - self.Q) / self.Q)
                    else:
                        self.bn += np.log((1 + self.Q) / (2 - self.Q))

                elif np.isclose(self.bn, -wn, atol=1e-8):
                    if action:
                        self.bn += np.log((2 - self.Q) / (1 + self.Q))
                    else:
                        self.bn += np.log(self.Q / (1 - self.Q))

                elif -wn < self.bn < wn:
                    if action:
                        self.bn += np.log((1 - self.Q) / self.Q)
                    else:
                        self.bn += np.log(self.Q / (1 - self.Q))

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
                    self.alpha = (1 + self.Q) * old_alpha / 2
                    self.beta = (2 - self.Q) * old_beta / 2
                elif np.isclose(self.bn0, -wn, atol=1e-8):
                    self.alpha = self.Q * old_alpha / 2
                    self.beta = (1 - self.Q) * old_beta / 2
                elif -wn < self.bn0 < wn:
                    self.alpha = self.Q * old_alpha
                    self.beta = (1 - self.Q) * old_beta
                elif self.bn0 > wn:
                    self.alpha = old_alpha
                    self.beta = old_beta
                elif self.bn0 < -wn:
                    self.alpha = self.beta = 0

                # second alpha beta update
                if np.isclose(self.bn1, wn, atol=1e-8):
                    self.alpha += (1 + self.Q) * (1 - old_alpha) / 2
                    self.beta += (2 - self.Q) * (1 - old_beta) / 2
                elif np.isclose(self.bn1, -wn, atol=1e-8):
                    self.alpha += self.Q * (1 - old_alpha) / 2
                    self.beta += (1 - self.Q) * (1 - old_beta) / 2
                elif -wn < self.bn1 < wn:
                    self.alpha += self.Q * (1 - old_alpha)
                    self.beta += (1 - self.Q) * (1 - old_beta)
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

    def _precompute_mc_actions(self):
        """
        Precomputes Monte Carlo actions, allowing agents inside the simulation
        to act as Perfect Bayesians by evaluating the empirical distribution
        of the MC paths dynamically. (Fully Vectorized)
        """
        actions = np.zeros(self.M, dtype=int)

        # We will use this to flatten 2D coordinates into 1D indices
        MAX_N = self.N + 1
        MAX_FLAT_INDEX = MAX_N ** 2

        for k in range(self.N):
            if k == 0:
                # Agent 0 has no neighbors, social LLR is 0
                total_llr = self.M_priv_llr[:, k]
            else:
                # 1. Generate stochastic neighborhoods for all M paths
                if self.graph_type == "ER":
                    num_ones = self.rng.binomial(self.M_running_ones, self.p)
                    num_zeros = self.rng.binomial(k - self.M_running_ones,
                                                  self.p)
                    num_neighbors = num_ones + num_zeros
                elif self.graph_type == "BS":
                    num_neighbors = np.full(self.M, min(k, self.sample))
                    num_ones = self.rng.hypergeometric(
                        self.M_running_ones,
                        k - self.M_running_ones,
                        num_neighbors
                    )

                # 2. Flatten the (num_neighbors, num_ones)
                # pairs into a 1D index
                flat_indices = (num_neighbors * MAX_N) + num_ones

                # Split indices by State 0 and State 1
                flat_idx_0 = flat_indices[:self.halfM]
                flat_idx_1 = flat_indices[self.halfM:]

                # 3. Fast counting using bincount
                # (equivalent to the dict counting)
                counts_0 = np.bincount(flat_idx_0, minlength=MAX_FLAT_INDEX)
                counts_1 = np.bincount(flat_idx_1, minlength=MAX_FLAT_INDEX)

                # Map the counts back to all M paths simultaneously
                c0_all = counts_0[flat_indices]
                c1_all = counts_1[flat_indices]

                # 4. Assign true Bayesian LLR
                # based on empirical MC distribution
                eps = 1e-10
                k_soc_llr = np.log((c0_all + eps) / (c1_all + eps))

                total_llr = self.M_priv_llr[:, k] + k_soc_llr

            # 5. Simulated agents make decisions based on True Bayesian LLR
            actions.fill(0)
            actions[total_llr < 0] = 1

            zero_filter = np.isclose(total_llr, 0, atol=1e-8)
            num_zero = np.sum(zero_filter)
            if num_zero:
                actions[zero_filter] = self.rng.choice([0, 1], size=num_zero)

            self.M_actions[:, k] = actions
            self.M_running_ones += actions

    def _mc_soc_llr(self, n, nbd, obs, history_array):
        """
        Computes social LLR using the Sufficient Statistic Approximation.
        Compresses the exact neighborhood vector into a simple sum.
        """
        # The sufficient statistic: How many 1s were observed?
        num_ones = np.sum(obs)

        # Extract the simulated actions for this specific neighborhood
        simulated_obs = self.M_actions[:, nbd]

        # Compress the simulated sequences into simulated sums
        simulated_ones = np.sum(simulated_obs, axis=1)

        # Count exact sufficient statistic matches in Theta=0 and Theta=1 paths
        count0 = np.sum(simulated_ones[:self.halfM] == num_ones)
        count1 = np.sum(simulated_ones[self.halfM:] == num_ones)

        # True Bayesian likelihood ratio of the sufficient statistic
        eps = 1e-10
        return np.log((count0 + eps) / (count1 + eps))
