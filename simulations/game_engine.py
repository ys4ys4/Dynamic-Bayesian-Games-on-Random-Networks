import numpy as np
import networkx as nx
import itertools
from scipy.stats import norm
from scipy.special import log_ndtr, logsumexp


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
                 rng=None,
                 signal_type="unbounded",
                 q=None,
                 k=None,
                 p=None,
                 sample=None,
                 M=None,
                 **kwargs
                 ):
        self.graph = graph
        self.graph_type = graph_type
        self.N = len(graph.nodes)
        self.rng = np.random.default_rng() if rng is None else rng
        self.signal_type = signal_type
        self.q = q if signal_type == "bounded" else norm.cdf(1)
        self.k = k
        self.p = p
        self.sample = sample
        self.M = M

        self.true_state = self.rng.choice([0, 1])
        self.history = np.zeros(self.N, dtype=int)
        self.played = False

        self.belief_engine = SequentialBeliefEngine(
            graph=self.graph,
            graph_type=self.graph_type,
            N=self.N,
            rng=self.rng,
            signal_type=self.signal_type,
            q=self.q,
            k=self.k,
            p=self.p,
            sample=self.sample,
            M=self.M
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
            bn = self.belief_engine.soc_llr(n, self.history)
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


class SequentialBeliefEngine:
    """
    creates a belief engine to compute posterior beliefs based on actions
    handles:
    computing private and social log-likelihood ratios
    updating beliefs based on actions
    """
    def __init__(self,
                 graph,
                 graph_type,
                 N,
                 rng=None,
                 signal_type="unbounded",
                 q=None,
                 k=None,
                 p=None,
                 sample=None,
                 M=None
                 ):
        self.adj_matrix = \
            nx.to_scipy_sparse_array(graph, format='csr').tocsr()
        self.graph_type = graph_type
        self.N = N
        # rng for Monte Carlo simulations in ER and BS graphs
        self.rng = np.random.default_rng() if rng is None else rng

        self.signal_type = signal_type
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

        self.mclist = ["ER", "BS"]

        if self.graph_type in self.mclist:
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

    def soc_llr(self, n, history):
        """
        computes social log-likelihood ratio for agent n based on history
        """
        if self.graph_type == "complete":
            return self.bn

        elif self.graph_type == "previous":
            if n == 0:
                return 0
            return self.bn1 if history[n - 1] else self.bn0

        ptr_start = self.adj_matrix.indptr[n]
        ptr_end = self.adj_matrix.indptr[n+1]
        if ptr_start == ptr_end:
            return 0
        nbd = self.adj_matrix.indices[ptr_start:ptr_end]
        obs = history[nbd]

        if self.graph_type in self.mclist:
            return self._mc_soc_llr(n, nbd, obs, history)

        num_ones = np.sum(obs)
        num_zeros = len(obs) - num_ones

        return (num_ones * np.log((1 - self.Q) / self.Q)) + \
            (num_zeros * np.log(self.Q / (1 - self.Q)))

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
        precompute Monte Carlo actions for all agents once so later queries
        only compare against stored trajectories.
        """

        k_soc_llr = np.zeros(self.M)
        actions = np.zeros(self.M, dtype=int)

        for k in range(self.N):
            if k:
                if self.graph_type == "ER":
                    num_ones = self.rng.binomial(self.M_running_ones, self.p)
                    num_zeros = self.rng.binomial(k - self.M_running_ones,
                                                  self.p)
                elif self.graph_type == "BS":
                    num_neighbours = min(k, self.sample)
                    num_ones = self.rng.hypergeometric(
                        self.M_running_ones,
                        k - self.M_running_ones,
                        num_neighbours
                    )
                    num_zeros = num_neighbours - num_ones

                k_soc_llr[:] = (num_ones * np.log((1 - self.Q) / self.Q))\
                    + (num_zeros * np.log(self.Q / (1 - self.Q)))
            total_llr = self.M_priv_llr[:, k] + k_soc_llr
            actions.fill(0)
            actions[total_llr < 0] = 1
            zero_filter = np.isclose(total_llr, 0, atol=1e-8)
            num_zero_filter = np.sum(zero_filter)
            if num_zero_filter:
                actions[zero_filter] = self.rng.choice([0, 1],
                                                       size=num_zero_filter)
            self.M_actions[:, k] = actions
            self.M_running_ones += actions

    def _mc_soc_llr(self, nbd, obs):
        """
        computes social log-likelihood ratio for agent n based on history
        using Monte Carlo simulation for ER and BS graphs
        """
        simulated_obs = self.M_actions[:, nbd]
        matches = np.all(simulated_obs == obs, axis=1)
        count0 = np.sum(matches[:self.halfM])
        count1 = np.sum(matches[self.halfM:])
        min_exact_matches = 0.005 * self.M
        if (count0 + count1) >= min_exact_matches:
            return np.log((count0 + 0.5) / (count1 + 0.5))

        distances = np.sum(simulated_obs != obs, axis=1)
        min_dist = np.min(distances)

        shifted_distances = distances - min_dist
        bandwidth = np.mean(shifted_distances)

        if bandwidth == 0:
            bandwidth = 1.0

        weights = np.exp(-shifted_distances / bandwidth)

        weight0 = np.sum(weights[:self.halfM])
        weight1 = np.sum(weights[self.halfM:])

        eps = 1e-10
        return np.log((weight0 + eps) / (weight1 + eps))


class RepeatedGame:
    """
    creates a repeated game with:
    graph = directed graph on which to play the game
    graph_type (str)
    signal_type = 'bounded' or 'unbounded'
    q = signal accuracy
    rng = random number generator for reproducibility

    handles:
    signal generation
    decision making
    game playing and tracking
    metrics for convergence and running accuracy
    """

    def __init__(self,
                 graph,
                 graph_type,
                 rng=None,
                 **kwargs
                 ):
        self.graph = graph
        self.graph_type = graph_type
        self.N = len(graph.nodes)
        self.rng = np.random.default_rng() if rng is None else rng
        if self.graph_type == "connected_star":
            self.max_T = 7
        else:
            self.max_T = 100

        self.true_state = self.rng.choice([0, 1])

        self.history = []
        self.played = False
        self.converged_at_t = None

        self.belief_engine = RepeatedBeliefEngine(
            graph=self.graph,
            graph_type=self.graph_type,
            N=self.N
        )

    def draw_signals(self):
        if self.true_state:
            return self.rng.normal(-1, 1, self.N)
        return self.rng.normal(1, 1, self.N)

    def decide(self, ans, bnts):
        total_llrs = ans + bnts
        actions = np.zeros(self.N, dtype=int)
        actions[total_llrs < 0] = 1
        zero_filter = np.isclose(total_llrs, 0, atol=1e-8)
        num_zero_filter = np.sum(zero_filter)
        if num_zero_filter:
            actions[zero_filter] = self.rng.choice([0, 1],
                                                   size=num_zero_filter)
        return actions

    def play(self):
        signals = self.draw_signals()
        ans = self.belief_engine.priv_llrs(signals)
        ans[0] = -ans[0]

        for t in range(self.max_T):
            bnts = self.belief_engine.soc_llrs(t, self.history)
            actions = self.decide(ans, bnts)
            self.history.append(actions)

            if np.all(actions == actions[0]):
                self.converged_at_t = t
                break

            self.belief_engine.update_beliefs(bnts, actions)

        self.played = True

    def convergence_metrics(self):
        if not self.played:
            return False, None, None, None

        final_actions = self.history[-1]
        final_accuracy = np.mean(final_actions == self.true_state)

        reached_absorbing_state = (self.converged_at_t is not None)

        is_consensus = np.all(final_actions == final_actions[0])
        success = bool(is_consensus and final_actions[0] == self.true_state)
        return (
            reached_absorbing_state,
            success,
            self.converged_at_t,
            final_accuracy,
        )

    def running_accuracy(self):
        accuracies = [
            np.mean(actions == self.true_state) for actions in self.history
        ]
        return np.array(accuracies)


class RepeatedBeliefEngine:
    """
    Creates a belief engine for repeated games.
    For Complete and Dictator Star graphs: uses exact global bounds updating.
    For Connected Star graphs: uses exact closed-form dynamic programming to
    perfectly marginalize over the hidden universes (Exact PBU).
    """
    def __init__(self, graph, graph_type, N):
        self.adj_matrix = nx.to_scipy_sparse_array(graph, format='csr').tocsr()
        self.graph_type = graph_type
        self.N = N

        self.lbs = np.full(self.N, -np.inf)
        self.ubs = np.full(self.N, np.inf)
        self.contribs = np.zeros(self.N)

        # caches for exact combinatorial marginalisation on connected star
        if self.graph_type == "connected_star":
            self.edge_cache = {}
            self.hub_cache = {}
            self.edge_bounds_cache = {}

    def priv_llrs(self, signals):
        return 2 * signals

    def soc_llrs(self, t, history):
        if self.graph_type == "complete_connected":
            return self.adj_matrix.dot(self.contribs)

        elif self.graph_type == "dictator_star":
            return self.adj_matrix.dot(self.contribs)

        elif self.graph_type == "connected_star":
            bnts = np.zeros(self.N)
            if t == 0:
                return bnts

            hist_matrix = np.array(history)
            H_h = tuple(hist_matrix[:, 0])

            # hub sees all edges: sum of their exact LLRs at end of t-1
            hub_bnt = 0.0
            hub_hist_for_edges = H_h[:t-1] if t >= 1 else ()
            for i in range(1, self.N):
                H_i = tuple(hist_matrix[:, i])
                hub_bnt += self._get_edge_llr(H_i, hub_hist_for_edges)
            bnts[0] = hub_bnt

            # each edge sees hub: subjective hub LLR from that edge's pov
            for i in range(1, self.N):
                H_i = tuple(hist_matrix[:, i])
                bnts[i] = self._get_hub_llr(H_i, H_h)

            return bnts

        return np.zeros(self.N)

    def update_beliefs(self, bnts, actions):
        # global bounds updating if not connected star
        if self.graph_type in ["complete_connected", "dictator_star"]:
            thresholds = -bnts / 2.0
            self.ubs = np.where(actions == 1, np.minimum(self.ubs, thresholds),
                                self.ubs)
            self.lbs = np.where(actions == 0, np.maximum(self.lbs, thresholds),
                                self.lbs)

            p_ts0 = norm.cdf(self.ubs - 1) - norm.cdf(self.lbs - 1)
            p_ts1 = norm.cdf(self.ubs + 1) - norm.cdf(self.lbs + 1)

            p_ts0 = np.clip(p_ts0, 1e-15, 1.0)
            p_ts1 = np.clip(p_ts1, 1e-15, 1.0)

            self.contribs = np.log(p_ts0 / p_ts1)

    # ---------------------------------------------------------
    # EXACT CONNECTED STAR MARGINALIZATION METHODS
    # ---------------------------------------------------------

    def _get_edge_bounds(self, H_s, H_h):
        """
        Calculates exact bounds for a edge.
        H_s: edge history of length k
        H_h: Hub history of length k-1 (the history that caused H_s)
        """
        k = len(H_s)
        if k == 0:
            return -np.inf, np.inf

        state = (H_s, H_h)
        if state in self.edge_bounds_cache:
            return self.edge_bounds_cache[state]

        prev_H_s = H_s[:-1]
        prev_H_h = H_h[:-1] if k - 1 > 0 else ()

        L, U = self._get_edge_bounds(prev_H_s, prev_H_h)

        # threshold for the kth action which
        # depends on subjective hub belief exactly before taking it
        bnt = self._get_hub_llr(prev_H_s, H_h)
        threshold = -bnt / 2.0

        action = H_s[-1]
        if action == 0:
            L = max(L, threshold)
        else:
            U = min(U, threshold)

        self.edge_bounds_cache[state] = (L, U)
        return L, U

    def _get_edge_llr(self, H_s, H_h):
        """
        Exact LLR contribution of a edge.
        H_s: edge history of length k
        H_h: Hub history of length k-1
        """
        k = len(H_s)
        if k == 0:
            return 0.0

        state = (H_s, H_h)
        if state in self.edge_cache:
            return self.edge_cache[state]

        L, U = self._get_edge_bounds(H_s, H_h)
        log_p0 = self._log_gaussian_interval_mass(L, U, 1)
        log_p1 = self._log_gaussian_interval_mass(L, U, -1)

        if np.isneginf(log_p0) and np.isneginf(log_p1):
            return 0.0

        llr = log_p0 - log_p1
        self.edge_cache[state] = llr
        return llr

    @staticmethod
    def _log_gaussian_interval_mass(lower, upper, mean):
        if upper <= lower:
            return -np.inf

        lower -= mean
        upper -= mean
        if lower >= 0:
            log_upper = log_ndtr(-upper)
            log_lower = log_ndtr(-lower)
        elif upper <= 0:
            log_upper = log_ndtr(upper)
            log_lower = log_ndtr(lower)
        else:
            log_upper = log_ndtr(upper)
            log_lower = log_ndtr(lower)
        log_mass, sign = logsumexp(
            [log_upper, log_lower],
            b=[-1.0, 1.0] if lower >= 0 else [1.0, -1.0],
            return_sign=True,
        )
        return log_mass if sign > 0 else -np.inf

    def _get_hub_llr(self, H_s, H_h):
        """
        Exact LLR of the Hub from the subjective perspective of a edge.
        H_s: edge's own history of length k
        H_h: Hub history of length k
        """
        k = len(H_h)
        if k == 0:
            return 0.0

        state = (H_s, H_h)
        if state in self.hub_cache:
            return self.hub_cache[state]

        log_totals_0 = []
        log_totals_1 = []

        # no. of actions taken by unseen edges up to this point
        num_unseen_actions = k - 1
        # hub history needed to evaluate an unseen
        # edge's history of length k-1 is length k-2
        hub_hist_for_unseen = H_h[:k-2] if k >= 2 else ()

        all_possible_edge_histories =\
            list(itertools.product([0, 1], repeat=num_unseen_actions))

        for other_histories in itertools.product(all_possible_edge_histories,
                                                 repeat=self.N - 2):
            log_prob_combo_0 = 0.0
            log_prob_combo_1 = 0.0

            # 1. what is prob of this specific unseen universe occurring?
            for H_other in other_histories:
                L_other, U_other = self._get_edge_bounds(H_other,
                                                         hub_hist_for_unseen)
                log_p0 = self._log_gaussian_interval_mass(L_other, U_other, 1)
                log_p1 = self._log_gaussian_interval_mass(L_other, U_other, -1)
                log_prob_combo_0 += log_p0
                log_prob_combo_1 += log_p1

            if (np.isneginf(log_prob_combo_0)
                    and np.isneginf(log_prob_combo_1)):
                continue

            # 2. reconstruct hub's exact threshold path in specific universe
            L_hub, U_hub = -np.inf, np.inf
            for tau in range(k):
                action = H_h[tau]

                # hub belief at time tau depends on edges' histories up to tau
                H_s_tau = H_s[:tau]
                H_h_tau_minus_1 = H_h[:tau-1] if tau >= 1 else ()

                sum_llrs = self._get_edge_llr(H_s_tau, H_h_tau_minus_1)

                for H_other in other_histories:
                    H_other_tau = H_other[:tau]
                    sum_llrs += self._get_edge_llr(H_other_tau,
                                                   H_h_tau_minus_1)

                threshold = -sum_llrs / 2.0
                if action == 0:
                    L_hub = max(L_hub, threshold)
                else:
                    U_hub = min(U_hub, threshold)

            # 3. what is prob that hub played H_h in this specific universe?
            log_hub_p0 = self._log_gaussian_interval_mass(L_hub, U_hub, 1)
            log_hub_p1 = self._log_gaussian_interval_mass(L_hub, U_hub, -1)

            # 4. integrate total weighted probability
            if not np.isneginf(log_hub_p0):
                log_totals_0.append(log_prob_combo_0 + log_hub_p0)
            if not np.isneginf(log_hub_p1):
                log_totals_1.append(log_prob_combo_1 + log_hub_p1)

        if not log_totals_0 and not log_totals_1:
            return 0.0

        llr = logsumexp(log_totals_0) - logsumexp(log_totals_1)
        self.hub_cache[state] = llr
        return llr
