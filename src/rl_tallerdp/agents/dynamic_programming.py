"""Programación dinámica para MDPs finitos.

Acepta dos formatos de P:
  - densa:   array (nS, nA, nS)           (MDPs pequeños, p. ej. Taxi-v3)
  - dispersa: scipy.sparse (nS * nA, nS)  (MDPs grandes, ver utils/mdp.py)
R siempre es un array (nS, nA).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import logging
log = logging.getLogger(__name__)


def policy_matrices(P, R, pi):
    """Average the dynamics over the policy: returns (P_pi, R_pi)."""
    R_pi = np.einsum("sa,sa->s", pi, R)
    if sp.issparse(P):
        nS, nA = R.shape
        selector = sp.csr_matrix(
            (pi.ravel(), (np.repeat(np.arange(nS), nA), np.arange(nS * nA))),
            shape=(nS, nS * nA),
        )
        return selector @ P, R_pi
    return np.einsum("sa,sat->st", pi, P), R_pi


def solve_policy_direct(P, R, pi, gamma):
    """Exact V^pi by solving (I - gamma P^pi) V = R^pi."""
    P_pi, R_pi = policy_matrices(P, R, pi)
    if sp.issparse(P_pi):
        A = (sp.identity(P_pi.shape[0], format="csc") - gamma * P_pi).tocsc()
        return spla.spsolve(A, R_pi)
    I = np.eye(P_pi.shape[0])
    return np.linalg.solve(I - gamma * P_pi, R_pi)


def iterative_policy_evaluation(P, R, pi, gamma, tol=1e-10, max_iter=100_000):
    P_pi, R_pi = policy_matrices(P, R, pi)
    V = np.zeros(R.shape[0])
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
    if sp.issparse(P):
        return R + gamma * (P @ V).reshape(R.shape)
    return R + gamma * (P @ V)


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
