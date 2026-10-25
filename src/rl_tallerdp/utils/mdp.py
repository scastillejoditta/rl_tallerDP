"""Construcción de las matrices (P, R) del MDP a partir de un entorno con `env.P`.

P es una matriz dispersa apilada de forma (nS * nA, nS): la fila s * nA + a
contiene la distribución de siguiente estado al tomar la acción a en s.
R es densa de forma (nS, nA) con la recompensa esperada E[r | s, a].

Estados terminales: las transiciones con done=True aportan su recompensa a R
pero NO tienen masa en P (el valor del estado terminal es 0).
"""
import numpy as np
import scipy.sparse as sp
import logging
log = logging.getLogger(__name__)

def build_sparse_mdp(env):
    nS, nA = env.nS, env.nA
    rows, cols, vals = [], [], []
    R = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for prob, ns, r, done in env.P[s][a]:
                R[s, a] += prob * r
                if not done:
                    rows.append(s * nA + a)
                    cols.append(ns)
                    vals.append(prob)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(nS * nA, nS))
    return P, R
