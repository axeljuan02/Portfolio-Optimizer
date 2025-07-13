# Portfolio-Optimizer

**Portfolio-Optimizer** is a personal project aligned with my finance ambitions, leveraging Harry Markowitz’s **Modern Portfolio Theory (MPT)** to optimize diversified investment portfolios, including cryptocurrencies, stocks, ETFs, and more.


## ✨ Features

- 🎯 Multi-asset portfolio support with customizable tickers and allocations  
- 📊 Historical market data via `yfinance` over user-defined timeframes  
- 📈 Calculation of `log-returns`, `covariance matrix`, `portfolio volatility`, `expected returns`, `beta (vs. SPY)`, and `Sharpe ratio`  
- ⚙️ Three optimization strategies with constraints:  
  1. Maximize Sharpe Ratio  
  2. Maximize Expected Return  
  3. Minimize Volatility  
- 🔒 Practical constraints (customizable) : weights sum to 1, no short selling, max 50% per asset  


## 🛠️ Tech Stack

`Python` | `pandas` | `numpy` | `scipy.optimize` | `yfinance`


## 🛤️ Roadmap

| Feature                         | Status        |
| ------------------------------- | ------------- |
| Streamlit integration           | 🚧 In Progress |
| Full website frontend (HTML/CSS/JS) | Planned  |
| Expanded asset classes support  | Planned       |
| Advanced risk metrics           | Planned       |
