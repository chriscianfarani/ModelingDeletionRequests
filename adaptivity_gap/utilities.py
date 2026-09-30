from itertools import permutations
from scipy.stats import multinomial, poisson

def nonadaptive_util(rv, n: int, order: list[int], k: int) -> float:
    if len(rv.mean()) != 3 or len(order) != 3:
        raise NotImplementedError('Only implemented for m=3')

    p: list[float] = [i/n for i in rv.mean()]

    p_a: float = p[order[0]] * (rv.cdf(k)[order[0]] - rv.cdf(0)[order[0]])

    prob: float = 0.
    for i in range(1,k+1):
        prob += rv.pmf(i)[order[1]] * rv.cdf(k-i)[order[0]]
    p_b: float = p[order[1]] * prob

    prob: float = 0.
    for i in range(1,k+1):
        for j in range(k+1-i):
            prob += rv.pmf(i)[order[2]] * rv.pmf(j)[order[0]] * rv.cdf(k-i-j)[order[1]]
    p_c: float = p[order[2]] * prob
    return p_a + p_b + p_c

def adaptive_util(rv, n: int, order: list[int], k: int, t: int) -> float:
    if len(rv.mean()) != 3 or len(order) != 3:
        raise NotImplementedError('Only implemented for m=3')

    p: list[float] = [i/n for i in rv.mean()]

    p_a: float = p[order[0]] * (rv.cdf(k)[order[0]] - rv.cdf(0)[order[0]])

    # p_a_t: float = rv.cdf(t)[order[0]]

    prob: float = 0.
    for i in range(1,k+1):
        prob += rv.pmf(i)[order[1]] * (rv.cdf(k-i)[order[0]] - rv.cdf(min(k-i,t))[order[0]])
    p_b1: float = p[order[1]] * prob

    prob: float = 0.
    for i in range(1,k+1):
        for j in range(t+1,k+1-i):
            prob += rv.pmf(i)[order[2]] * rv.pmf(j)[order[0]] * rv.cdf(k-i-j)[order[1]]
    p_c1: float = p[order[2]] * prob

    prob: float = 0.
    for i in range(1,k+1):
        prob += rv.pmf(i)[order[2]] * (rv.cdf(min(k-i,t))[order[0]])
    p_c2: float = p[order[2]] * prob

    prob: float = 0.
    for i in range(1,k+1):
        for j in range(t+1):
            prob += rv.pmf(i)[order[1]] * rv.pmf(j)[order[0]] * rv.cdf(k-i-j)[order[2]]
    p_b2: float = p[order[1]] * prob
    # return p_a + (1 - p_a_t) * (p_b1 + p_c1) + p_a_t * (p_b2 + p_c2)
    return p_a + (p_b1 + p_c1) + (p_b2 + p_c2)

def adaptivity_gap(n: int, k: int, p: list[float]) -> float:
    rv = poisson([n*i for i in p])
    max_nonadaptive: float = max([nonadaptive_util(rv, n, order, k) for order in permutations(range(3))])
    max_adaptive: float = max([max([adaptive_util(rv, n, order, k, t) for t in range(k)]) for order in permutations(range(3))])
    return max_adaptive / max_nonadaptive