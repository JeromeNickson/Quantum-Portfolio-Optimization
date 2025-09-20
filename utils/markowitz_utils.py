import numpy as np
import cvxpy as cp

def run_markowitz(mu, Sigma, rf=0.02, n_assets=None, target_points=50):
    """
    Computes the Markowitz tangency portfolio (max Sharpe) and efficient frontier.

    Parameters:
    mu          : np.array, expected returns
    Sigma       : np.array, covariance matrix
    rf          : float, risk-free rate
    n_assets    : int, number of assets (default: len(mu))
    target_points : int, number of points on efficient frontier

    Returns:
    tangency_weights : np.array, optimal weights (max Sharpe)
    tangency_return  : float, expected return
    tangency_risk    : float, volatility
    risks            : list of volatilities for frontier
    rets             : list of returns for frontier
    """
    if n_assets is None:
        n_assets = len(mu)

    # Function to solve for a given target return
    def solve_markowitz_target(target_return):
        w = cp.Variable(n_assets)
        portfolio_return = mu @ w
        portfolio_risk = cp.quad_form(w, Sigma)
        prob = cp.Problem(cp.Minimize(portfolio_risk),
                          [cp.sum(w) == 1, w >= 0, portfolio_return >= target_return])
        prob.solve()
        if prob.status == "optimal":
            return w.value, portfolio_return.value, portfolio_risk.value
        else:
            return None, None, None

    # Efficient frontier
    target_returns = np.linspace(min(mu), max(mu), target_points)
    risks, rets, weights = [], [], []

    for r in target_returns:
        w_opt, pret, prisk = solve_markowitz_target(r)
        if w_opt is not None:
            risks.append(np.sqrt(prisk))
            rets.append(pret)
            weights.append(w_opt)

    # Tangency (max Sharpe) portfolio
    sharpe_ratios = [(r - rf)/s if s>0 else -1 for r,s in zip(rets, risks)]
    max_idx = np.argmax(sharpe_ratios)

    tangency_weights = weights[max_idx]
    tangency_return = rets[max_idx]
    tangency_risk = risks[max_idx]

    return tangency_weights, tangency_return, tangency_risk, risks, rets

