import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load expected returns and covariance matrix
mu = pd.read_csv("nifty100_expected_returns.csv", index_col=0).squeeze()  # Series
cov = pd.read_csv("nifty100_covariance_matrix.csv", index_col=0)  # DataFrame

# Match columns (in case some tickers were dropped)
tickers = list(set(mu.index).intersection(cov.columns))
mu = mu[tickers].values
cov = cov.loc[tickers, tickers].values
n = len(tickers)

print(f"✅ Optimizing for {n} stocks...")

# Store risk-return points
risks, returns = [], []

# Try multiple lambda values to get Pareto frontier
lambdas = np.linspace(0.1, 0.9, 9)

for lam in lambdas:
    # Variables
    w = cp.Variable(n)            # portfolio weights
    z = cp.Variable(n, boolean=True)  # selection variable

    # Objective: weighted return - weighted risk
    expected_return = mu @ w
    risk = cp.quad_form(w, cov)
    objective = cp.Maximize(lam * expected_return - (1 - lam) * risk)

    # Constraints
    constraints = [
        cp.sum(w) == 1,                              # fully invested
        cp.sum(z) >= 10,                             # at least 10 assets
        cp.sum(z) <= 15,                             # at most 15 assets
        w >= 0.05 * z,                               # min weight per selected stock
        w <= 0.20 * z                                # max weight per selected stock
    ]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI if cp.GUROBI in cp.installed_solvers() else cp.SCIP if cp.SCIP in cp.installed_solvers() else cp.ECOS_BB)

    if w.value is not None:
        portfolio_return = mu @ w.value
        portfolio_risk = np.sqrt(w.value @ cov @ w.value)
        risks.append(portfolio_risk)
        returns.append(portfolio_return)

        # Print top assets for each lambda
        selected_assets = [tickers[i] for i in range(n) if z.value[i] > 0.5]
        print(f"\nλ = {lam:.1f}")
        print("Selected Stocks:", selected_assets)
        print("Portfolio Return:", portfolio_return)
        print("Portfolio Risk:", portfolio_risk)
    else:
        print(f"\nλ = {lam:.1f} → Optimization failed.")

# Plot efficient frontier
plt.figure(figsize=(8, 6))
plt.plot(risks, returns, marker='o', color='purple')
plt.title("Efficient Frontier (ILP with Cardinality + Min/Max Constraints)")
plt.xlabel("Portfolio Risk (Std. Dev.)")
plt.ylabel("Expected Return")
plt.grid(True)
plt.show()
