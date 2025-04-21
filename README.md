# 📊 NSGA-II Based Portfolio Optimizer using Nifty 100

This project is a Streamlit-based portfolio optimization application built with a custom implementation of **NSGA-II (Non-dominated Sorting Genetic Algorithm II)**. It helps users find the best trade-offs between **risk** and **return**, using historical data from **Nifty 100** stocks.

---

## 🚀 Features

- ⚙️ **Custom NSGA-II implementation** (no external optimization libraries)
- 📈 **Multi-objective optimization**: maximize return, minimize risk
- 💼 User-defined **capital** and **risk tolerance**
- 📊 **Pareto front visualization** of optimal portfolios
- 📂 Upload custom return and covariance matrix files
- ✅ Filters portfolios based on user’s maximum risk level
- 🧠 Recommends an investment portfolio within risk preferences

---

## 📁 File Structure

├── app.py # Main Streamlit application ├── requirements.txt # Dependencies ├── README.md # You're reading it! ├── data/ │ ├── expected_returns.csv │ └── covariance_matrix.csv


---

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nifty100-portfolio-optimizer.git
   cd nifty100-portfolio-optimizer

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
📤 Inputs Required
expected_returns.csv: 1-year average returns of selected Nifty 100 stocks

covariance_matrix.csv: Covariance matrix for the same stocks

Example files are available in the data/ folder.

🧠 Behind the Scenes
This project uses a custom-built NSGA-II algorithm for solving a two-objective optimization problem:

Objective 1: Minimize portfolio variance (risk)

Objective 2: Maximize portfolio return

We use:

Expected returns vector 
𝜇
μ

Covariance matrix 
Σ
Σ

Portfolio weights 
𝑤
w satisfying 
∑
𝑤
=
1
∑w=1

Final portfolios are filtered based on:

Total capital

User-defined maximum acceptable risk level

📷 Visuals
📍 Pareto Front
Shows optimal trade-offs between return and risk.

📈 Portfolio Allocation
Bar chart showing how much to invest in each selected stock.

📌 Future Work
Incorporate real-time stock data (via APIs)

Add constraints like sector diversity, ESG filters

Include ILP and comparison with NSGA-II

Improve performance with parallel processing

🧑‍💻 Author
Akash Deepak Varma Vaddi
B.Tech, IIITDM Chennai
Email: [akashdeepakvarma.gamil.com]

