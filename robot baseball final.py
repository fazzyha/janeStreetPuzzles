# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 17:05:44 2025

@author: farre

soln for jane street puzzle oct '25 - robot baseball
"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict

MAX_BALLS = 4
MAX_STRIKES = 3
PURE_EPS = 1e-15 # tolerance for pure saddle: : min_row_max ≈ max_col_min
DEGEN_EPS = 1e-15

@dataclass
class StateResult:
    V: float # expected runs for batter from current (b,s)
    R: float # prob of reaching full count from current (b,s)
    x: float # batter's mix P(Sw). 1-P(Sw) = P(W)
    y: float # pitcher's mix P(Str). 1-P(Str) = P(B)


def solve_full_game(p: float) -> Dict[Tuple[int, int], StateResult]:
    """
    solves for fixed p via backward induction over (b,s)
    Return: a dict mapping (b,s) - StateResult with V (expected runs from here)
    , R (prob of reaching full count), mixed startegies (x,y)
    """
    # define V&R for all states, including match end
    V = {(b, s): 0.0 for b in range(5) for s in range(4)}
    R = {(b, s): 0.0 for b in range(5) for s in range(4)}
    X = {(b, s): 0.0 for b in range(5) for s in range(4)}
    Y = {(b, s): 0.0 for b in range(5) for s in range(4)}
    
    #walk: b=4. 1 Run, but R=0 (not via full count)
    for s in range(3):
        V[(4, s)] = 1.0
        R[(4, s)] = 0.0
        
    # same for strikeout (s=3). 0 runs
    for b in range(4):
        V[(b, 3)] = 0.0
        R[(b, 3)] = 0.0
    
    # full count indicator:
    R[(3, 2)] = 1.0

    def solve_cell(A: float, B: float, C: float) -> Tuple[float, float, float]:
        """
        Return (V_cell, x, y) for the subgame with payoffs [ [A,B],[B,C] ].
        """
        # payoff matrix:
        #           Batter: Wait    Batter: Swing
        # P:Ball   [     A         ,      B       ]
        # P:Strike [     B         ,      C       ]
        row_max0 = max(A, B)
        row_max1 = max(B, C)
        min_row_max = min(row_max0, row_max1)

        col_min0 = min(A, B)
        col_min1 = min(B, C)
        max_col_min = max(col_min0, col_min1)

        if abs(min_row_max - max_col_min) <= PURE_EPS: # pure saddle point
            val = min_row_max
            candidates = [((0,0), A), ((0,1), B), ((1,0), B), ((1,1), C)]
            for (i, j), pay in candidates: # finds pure action pairs
                if abs(pay - val) <= PURE_EPS:
                    x = float(j) # batter
                    y = float(i) # pitcher
                    return val, x, y

        # mixed-strategy case. No pure saddle
        D = A - 2*B + C
        if abs(D) <= DEGEN_EPS: # lines coincide. unstable
            if row_max0 <= row_max1: # find sensible pure outcome
                if A >= B:
                    return A, 0.0, 0.0
                else:
                    return B, 1.0, 0.0
            else:
                if C >= B:
                    return C, 1.0, 1.0
                else:
                    return B, 0.0, 1.0

        # symmetric off-diagonal - so same formula!
        x = (A - B) / D
        y = (A - B) / D
        # stop floating point errors leaking
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        # state's value
        V_cell = (A*C - B*B) / D
        return V_cell, x, y

    # backward pass. so states forward are already known
    for b in range(3, -1, -1):
        for s in range(2, -1, -1):
            if (b, s) == (3, 2): # no need to recompute R at full count
                A = V[(4, 2)]
                B = V[(3, 3)]
                C = 4*p + (1-p)*V[(3, 3)]
                v, x, y = solve_cell(A, B, C)
                V[(b, s)], X[(b, s)], Y[(b, s)] = v, x, y
                continue

            A = V[(b+1, s)] # otherwise fill to arbitrary next state
            B = V[(b, s+1)]
            C = 4*p + (1-p) * V[(b, s+1)]
            v, x, y = solve_cell(A, B, C)
            V[(b, s)] = v
            X[(b, s)] = x
            Y[(b, s)] = y

            # state `transition probablities
            pw_ball_wait   = (1 - y) * (1 - x)
            pw_ball_swing  = (1 - y) * x
            pw_strike_wait = y * (1 - x)
            pw_strike_swing = y * x

            prob_to_b_splus1 = pw_ball_swing + pw_strike_wait + pw_strike_swing * (1 - p)
            prob_to_bplus1_s = pw_ball_wait

            R[(b, s)] = (
                prob_to_bplus1_s * R[(b+1, s)] +
                prob_to_b_splus1 * R[(b, s+1)]
            )

    out = {(b, s): StateResult(V[(b, s)], R[(b, s)], X[(b, s)], Y[(b, s)])
           for b in range(4) for s in range(3)}
    out[(3, 2)] = StateResult(V[(3, 2)], R[(3, 2)], X[(3, 2)], Y[(3, 2)])
    # explictly include full count
    return out

def q_of_p(p: float) -> float:
    # reach full count prob from solving from beginning
    res = solve_full_game(p)
    return res[(0, 0)].R

def maximize_q_precise() -> Tuple[float, float]:
    """
    maximize q(p) over [0,1] using:
    - dense grid to find a bracket [a,b],
    , then Brent-style 1D optimisation on f(p) = −q(p)
    """
    N = 4001 # dense but computable grid
    grid = [i/(N-1) for i in range(N)]
    vals = [q_of_p(p) for p in grid]
    k = max(range(N), key=lambda i: vals[i])

    # expand bracket around grid max
    a_idx = max(0, k-1)
    b_idx = min(N-1, k+1)
    while a_idx > 0 and vals[a_idx-1] >= vals[a_idx]:
        a_idx -= 1
    while b_idx < N-1 and vals[b_idx+1] >= vals[b_idx]:
        b_idx += 1
    # covers local rise around peak

    a = grid[a_idx]
    b = grid[b_idx]
    if a == b: # for Brent to start non-zero
        a = max(0.0, grid[k] - 0.01)
        b = min(1.0, grid[k] + 0.01)

    def f(p): return -q_of_p(p) # maximise q (min -q)

    eps = 1e-13
    tol = 1e-12
    x = w = v = 0.5*(a+b) # x: current best minimiser estimate
    # w,v previous parabolic interpolation steps
    fx = fw = fv = f(x) #fn at these points
    d = e = b - a # ult and penult step sizes

    while True:
        m = 0.5*(a+b)
        tol1 = math.sqrt(eps)*abs(x) + tol # dynamic tolerance with tol to avoid infinite
        tol2 = 2*tol1
        if abs(x - m) <= tol2 - 0.5*(b - a): # if x close enough compared to bracket mid - stop
            p_star = x
            q_star = -fx
            return p_star, q_star

        # try parabolic step if prev not small enough
        p_step = False
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p_par = (x - v) * q - (x - w) * r
            q_par = 2.0 * (q - r)
            # fit parabola and test new point
            if q_par > 0:
                p_par = -p_par # to MINIMISE
            q_par = abs(q_par)
            if (abs(p_par) < abs(0.5*q_par*e) and # smaller than half prev step
                p_par > q_par*(a - x) and # within brakcet
                p_par < q_par*(b - x)): # "
                d = p_par / q_par
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if x < m else -tol1 # nudge if too close
                p_step = True # was parabolic

        if not p_step: # take golden section step
            invphi = (3.0 - math.sqrt(5.0)) / 2.0
            e = d = (b - a) * invphi # shrink by invphi
            d = -d if x >= m else d # move in more likely direction
            u = x + d
            if abs(u - x) < tol1: # if step too small, force min sized movement
                d = tol1 if (u - x) >= 0 else -tol1
                u = x + d

        #evaluate
        fu = f(u)
        
        # update points if u better than x. otherwise if u worse
        if fu <= fx:
            if u < x: b = x
            else:     a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
            e = d
        else:
            if u < x: a = u
            else:     b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu


if __name__ == "__main__":
    p_star, q_star = maximize_q_precise()
    print(f"p* = {p_star:.15f}")
    print(f"q* = {q_star:.15f}")
    print(f"q* (10dp) = {q_star:.10f}")

    # sanity checks
    print("\nLocal neighbor check around p*:")
    for dp in [-1e-6, -1e-7, -1e-8, 0.0, 1e-8, 1e-7, 1e-6]:
        p = max(0.0, min(1.0, p_star + dp))
        print(f"p = {p:.15f}  q(p) = {q_of_p(p):.15f}")


