import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.repair import Repair

st.set_page_config(layout="wide")
st.title("📊 NSGA-II Portfolio Optimizer - Nifty 100")

# =========================
# Upload Data
# =========================
st.sidebar.header("Step 1: Upload Data")
returns_file = st.sidebar.file_uploader("Upload expected returns CSV", type=["csv"])
cov_file = st.sidebar.file_uploader("Upload covariance matrix CSV", type=["csv"])

if returns_file and cov_file:
    mu = pd.read_csv(returns_file, index_col=0).squeeze()
    cov = pd.read_csv(cov_file, index_col=0)

    tickers = list(set(mu.index).intersection(cov.columns))
    mu = mu[tickers].values
    cov = cov.loc[tickers, tickers].values
    n_assets = len(tickers)

    st.success(f"Loaded data for {n_assets} assets.")

    # =========================
    # User Inputs
    # =========================
    st.sidebar.header("Step 2: Set Preferences")
    total_capital = st.sidebar.number_input("Total Capital (₹)", min_value=10000, value=1000000, step=10000)
    max_risk = st.sidebar.slider("Maximum Acceptable Risk (%)", min_value=1, max_value=50, value=15) / 100

    # =========================
    # Custom Repair
    # =========================
    class PortfolioRepair(Repair):
        def _do(self, problem, X, **kwargs):
            for i in range(len(X)):
                w = X[i]
                w[w < 0] = 0
                idx = np.argsort(-w)
                k = np.random.randint(10, 16)
                selected = idx[:k]
                w[:] = 0
                w[selected] = np.random.uniform(0.05, 0.2, size=k)
                w[selected] /= w[selected].sum()
            return X

    # =========================
    # Problem Definition
    # =========================
    class PortfolioOptimization(Problem):
        def __init__(self):
            super().__init__(n_var=n_assets, n_obj=2, n_constr=0, xl=0.0, xu=0.2)

        def _evaluate(self, X, out, *args, **kwargs):
            returns = X @ mu
            risks = np.einsum("ij,jk,ik->i", X, cov, X)
            out["F"] = np.column_stack([risks, -returns])

    # =========================
    # Run Optimization
    # =========================
    st.sidebar.header("Step 3: Optimize")
    if st.sidebar.button("Run NSGA-II Optimization"):
        with st.spinner("Running NSGA-II optimization..."):
            problem = PortfolioOptimization()
            algorithm = NSGA2(
                pop_size=150,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True,
                repair=PortfolioRepair()
            )
            termination = get_termination("n_gen", 200)
            res = minimize(problem, algorithm, termination, seed=42, verbose=False)

        F = res.F
        X = res.X
        returns = -F[:, 1]
        risks = np.sqrt(F[:, 0])

        # Pareto Plot
        st.subheader("Pareto Front")
        fig, ax = plt.subplots()
        ax.scatter(risks, returns, c='purple')
        ax.set_xlabel("Risk (Std Deviation)")
        ax.set_ylabel("Expected Return")
        ax.set_title("Pareto Front")
        st.pyplot(fig)

        # Filter portfolios based on risk
        valid_idxs = np.where(risks <= max_risk)[0]
        if len(valid_idxs) == 0:
            st.error("❌ No portfolio found within your risk tolerance.")
        else:
            st.success(f"✅ Found {len(valid_idxs)} portfolio(s) within your risk tolerance.")
            top_idx = valid_idxs[np.argmax(returns[valid_idxs])]
            weights = X[top_idx]
            selected = np.where(weights > 1e-3)[0]

            st.subheader("📊 Recommended Portfolio")
            st.markdown(f"- **Expected Return:** {returns[top_idx]:.2%}")
            st.markdown(f"- **Risk:** {risks[top_idx]:.2%}")
            st.markdown(f"- **Expected Profit:** ₹{returns[top_idx]*total_capital:,.0f}")

            portfolio_df = pd.DataFrame({
                'Stock': [tickers[i] for i in selected],
                'Weight %': [weights[i] * 100 for i in selected],
                'Investment ₹': [weights[i] * total_capital for i in selected]
            })
            st.dataframe(portfolio_df.set_index("Stock"))

            # Plot Allocation
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.bar(portfolio_df['Stock'], portfolio_df['Investment ₹'], color='teal')
            ax2.set_ylabel("Investment Amount (₹)")
            ax2.set_title("Recommended Portfolio Allocation")
            plt.xticks(rotation=90)
            st.pyplot(fig2)

else:
    st.info("👈 Upload both return and covariance CSV files to begin.")
