import numpy as np

def policy_matrices(P, R, pi):
    """Average the dynamics over the policy: returns (P_pi, R_pi)."""
    P_pi = np.einsum("sa,sat->st", pi, P)
    R_pi = np.einsum("sa,sa->s", pi, R)
    return P_pi, R_pi

def solve_policy_direct(P, R, pi, gamma):
    """Exact V^pi by solving (I - gamma P^pi) V = R^pi."""
    P_pi, R_pi = policy_matrices(P, R, pi)
    I = np.eye(P_pi.shape[0])
    V = np.linalg.solve(I - gamma * P_pi, R_pi)
    return V

def iterative_policy_evaluation(P, R, pi, gamma, tol=1e-10, max_iter=100_000):
    P_pi, R_pi = policy_matrices(P, R, pi)
    V = np.zeros(P.shape[0])
    deltas = []

    for k in range(max_iter):
        V_new = R_pi + gamma * P_pi @ V
        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        V = V_new
        if delta < tol:
            break
    
    return V, k + 1, deltas

def q_from_v(P, R, V, gamma):
    """One-step lookahead: Q[s, a] from V."""
    Q = R + gamma * (P @ V)
    return Q

def greedy_policy(Q):
    """Deterministic greedy policy, as a one-hot matrix."""
    pi = np.zeros_like(Q)
    pi[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1.0
    return pi

def policy_iteration(P, R, gamma, tol=1e-9, max_iter=1_000):
    pi = np.ones(R.shape) / R.shape[1]      # política inicial: uniforme aleatoria
    V = None

    for k in range(max_iter):
        V_new = solve_policy_direct(P, R, pi, gamma)
        if V is not None and np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
        pi = greedy_policy(q_from_v(P, R, V, gamma))

    return V, pi, k + 1

def value_iteration(P, R, gamma, tol=1e-10, max_iter=100_000):
    V = np.zeros(R.shape[0])
    deltas = []

    for k in range(max_iter):
          Q = q_from_v(P, R, V, gamma)
          V_new = Q.max(axis=1)
          delta = np.max(np.abs(V_new - V))
          deltas.append(delta)
          V = V_new
          if delta < tol:
              break
    pi = greedy_policy(q_from_v(P, R, V, gamma))
    return V, pi, k + 1, deltas