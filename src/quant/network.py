"""JurisFinanceAI - Network Analysis Engine

Implements graph theory for financial networks:
- Systemic risk and contagion modeling
- Too-big-to-fail identification
- Correlation networks
- Centrality analysis
- Information flow (simplified Transfer Entropy concept)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class NetworkAnalyzer:
    """Financial network analysis using graph theory.

    Models systemic risk, contagion, and inter-connectedness
    of financial entities (banks, companies, assets).
    """

    def __init__(self):
        self.last_result = None

    def build_correlation_network(self, returns_matrix, threshold=0.3,
                                   asset_names=None) -> Dict:
        """Build correlation-based network.

        Nodes = assets, Edges = correlations above threshold.
        """
        R = np.array(returns_matrix, dtype=float)
        n = R.shape[1]
        if asset_names is None:
            asset_names = [f"A{i+1}" for i in range(n)]

        corr = np.corrcoef(R, rowvar=False)
        np.fill_diagonal(corr, 0)

        # Build adjacency matrix (edges where |corr| > threshold)
        adjacency = np.abs(corr) > threshold
        adjacency = adjacency.astype(float)

        # Degree centrality
        degrees = adjacency.sum(axis=1)
        strength = (np.abs(corr) * adjacency).sum(axis=1)

        # Eigenvector centrality (simplified - power iteration)
        eig_centrality = np.ones(n) / n
        for _ in range(100):
            new_centrality = adjacency @ eig_centrality
            norm = np.linalg.norm(new_centrality)
            if norm == 0: break
            eig_centrality = new_centrality / norm

        # Betweenness centrality (simplified)
        betweenness = np.zeros(n)
        for s in range(n):
            for t in range(n):
                if s == t: continue
                # BFS shortest paths
                visited = {s}
                queue = [(s, [s])]
                paths = []
                while queue:
                    node, path = queue.pop(0)
                    for neighbor in np.where(adjacency[node] > 0)[0]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [neighbor]
                            if neighbor == t:
                                paths.append(new_path)
                            else:
                                queue.append((neighbor, new_path))
                for path in paths:
                    for node in path[1:-1]:
                        betweenness[node] += 1

        n_paths = max(n * (n - 1), 1)
        betweenness /= n_paths

        # Clustering coefficient
        clustering = np.zeros(n)
        for i in range(n):
            neighbors = np.where(adjacency[i] > 0)[0]
            k = len(neighbors)
            if k < 2: continue
            triangles = 0
            for a_idx in range(len(neighbors)):
                for b_idx in range(a_idx + 1, len(neighbors)):
                    if adjacency[neighbors[a_idx], neighbors[b_idx]] > 0:
                        triangles += 1
            clustering[i] = 2 * triangles / (k * (k - 1))

        # Network density
        n_edges = adjacency.sum() / 2
        max_edges = n * (n - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        self.last_result = {
            "method": "Correlation Network Analysis",
            "n_nodes": n,
            "n_edges": int(n_edges),
            "density": float(density),
            "threshold": threshold,
            "adjacency_matrix": adjacency.tolist(),
            "correlation_matrix": corr.tolist(),
            "asset_names": asset_names,
            "degree_centrality": degrees.tolist(),
            "strength_centrality": strength.tolist(),
            "eigenvector_centrality": eig_centrality.tolist(),
            "betweenness_centrality": betweenness.tolist(),
            "clustering_coefficient": clustering.tolist(),
            "most_connected": asset_names[int(np.argmax(degrees))],
            "most_central": asset_names[int(np.argmax(eig_centrality))],
        }
        return self.last_result

    def simulate_contagion(self, initial_shock_node, adjacency,
                           shock_magnitude=0.1,
                           transmission_rate=0.3,
                           recovery_rate=0.05,
                           max_rounds=20) -> Dict:
        """Simulate financial contagion through a network.

        Models how a shock to one node spreads to connected nodes.
        Similar to SIR epidemiological model applied to finance.
        """
        adj = np.array(adjacency, dtype=float)
        n = len(adj)

        shocked = np.zeros(n)
        shocked[initial_shock_node] = shock_magnitude

        history = [shocked.copy().tolist()]
        total_loss = shock_magnitude

        for round_num in range(max_rounds):
            new_shocked = shocked.copy()
            for i in range(n):
                if shocked[i] > 0.001:
                    for j in range(n):
                        if adj[i, j] > 0 and shocked[j] < shocked[i]:
                            transmission = transmission_rate * adj[i, j] * shocked[i]
                            new_shocked[j] += transmission
                            total_loss += transmission
                    new_shocked[i] *= (1 - recovery_rate)
            shocked = new_shocked
            history.append(shocked.copy().tolist())

            if np.max(shocked) < 0.001:
                break

        self.last_result = {
            "method": "Contagion Simulation",
            "initial_shock_node": initial_shock_node,
            "shock_magnitude": shock_magnitude,
            "transmission_rate": transmission_rate,
            "recovery_rate": recovery_rate,
            "rounds": len(history) - 1,
            "total_system_loss": float(total_loss),
            "contagion_history": history,
            "nodes_affected": int(np.sum(shocked > 0.001)),
            "max_node_impact": float(np.max(shocked)),
            "systemic_risk_score": min(float(total_loss / (n * shock_magnitude)), 1.0) if shock_magnitude > 0 else 0,
        }
        return self.last_result

    def identify_systemic_importance(self, adjacency, returns_matrix=None) -> Dict:
        """Identify systemically important entities (Too-Big-To-Fail).

    Combines multiple centrality measures.
    """
        adj = np.array(adjacency, dtype=float)
        n = len(adj)

        degrees = adj.sum(axis=1)

        # Eigenvector centrality
        eig_c = np.ones(n) / n
        for _ in range(100):
            new_c = adj @ eig_c
            norm = np.linalg.norm(new_c)
            if norm == 0: break
            eig_c = new_c / norm

        # Vulnerability: how much removing this node disconnects the network
        vulnerability = np.zeros(n)
        for i in range(n):
            adj_without = adj.copy()
            adj_without[i, :] = 0
            adj_without[:, i] = 0
            # Check if graph is still connected
            reachable = set()
            queue = [0 if i != 0 else 1]
            while queue:
                node = queue.pop(0)
                if node in reachable: continue
                reachable.add(node)
                for neighbor in np.where(adj_without[node] > 0)[0]:
                    if neighbor not in reachable:
                        queue.append(neighbor)
            vulnerability[i] = (n - len(reachable)) / max(n - 1, 1)

        # Composite systemic importance score
        max_deg = max(degrees.max(), 1)
        composite = (degrees / max_deg * 0.3 +
                     eig_c * 0.3 +
                     vulnerability * 0.4)

        ranking = np.argsort(-composite)

        self.last_result = {
            "method": "Systemic Importance Identification",
            "degree_centrality": degrees.tolist(),
            "eigenvector_centrality": eig_c.tolist(),
            "vulnerability_index": vulnerability.tolist(),
            "composite_importance": composite.tolist(),
            "ranking": ranking.tolist(),
            "most_systemic": int(ranking[0]),
            "too_big_to_fail": [int(r) for r in ranking[:max(3, n // 5)]],
        }
        return self.last_result

    def information_flow_matrix(self, returns_matrix) -> Dict:
        """Simplified information flow / Granger-causality-inspired analysis.

    Measures directional influence between assets using
    lagged regression (simplified Transfer Entropy concept).
    """
        R = np.array(returns_matrix, dtype=float)
        n_assets = R.shape[1]
        n_obs = R.shape[0]

        flow = np.zeros((n_assets, n_assets))
        lag = min(5, n_obs // 10)

        for i in range(n_assets):
            for j in range(n_assets):
                if i == j: continue
                # Regress R_j[t] on R_j[t-lag] (baseline)
                y = R[lag:, j]
                X_base = R[:-lag, j].reshape(-1, 1)
                X_base = np.column_stack([np.ones(len(y)), X_base])
                beta_base = np.linalg.lstsq(X_base, y, rcond=None)[0]
                rss_base = np.sum((y - X_base @ beta_base) ** 2)

                # Add R_i[t-lag] as predictor
                X_full = np.column_stack([X_base, R[:-lag, i]])
                beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
                rss_full = np.sum((y - X_full @ beta_full) ** 2)

                # F-statistic
                df1 = 1
                df2 = len(y) - X_full.shape[1]
                if df2 <= 0 or rss_full == 0:
                    flow[i, j] = 0
                else:
                    f_stat = ((rss_base - rss_full) / df1) / (rss_full / df2)
                    flow[i, j] = max(0, f_stat)

        # Normalize
        max_flow = flow.max()
        if max_flow > 0:
            flow_norm = flow / max_flow
        else:
            flow_norm = flow

        # Identify leaders and followers
        out_flow = flow_norm.sum(axis=1)
        in_flow = flow_norm.sum(axis=0)
        net_flow = out_flow - in_flow

        self.last_result = {
            "method": "Information Flow Analysis",
            "flow_matrix": flow.tolist(),
            "normalized_flow": flow_norm.tolist(),
            "out_flow": out_flow.tolist(),
            "in_flow": in_flow.tolist(),
            "net_flow": net_flow.tolist(),
            "leaders": np.argsort(-net_flow)[:5].tolist(),
            "followers": np.argsort(net_flow)[:5].tolist(),
        }
        return self.last_result
