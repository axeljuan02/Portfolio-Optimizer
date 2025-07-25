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
| Efficient Frontier Chart        | 🚧 In Progress |
| VaR and CVaR calculus           | Planned       |
| Streamlit integration           | Planned       |
| Full website frontend (HTML/CSS/JS) | Planned  |
| Further Opimization             | Planned       |

## ⚙️ First Results
<img width="2336" height="1122" alt="image" src="https://github.com/user-attachments/assets/e728f627-8bde-4398-bd73-4e5fc54d5d75" />


## ⚙️ How it works
<img width="650" height="300" alt="image" src="https://github.com/user-attachments/assets/2fabe7d0-308b-434d-bd9c-7b9a281d8f66" />
<img width="650" height="300" alt="portfolio volatility formula (stdev)" src="https://github.com/user-attachments/assets/fe435449-3f8c-4041-bc4f-aa7b43f9afda" />
<img width="650" height="300" alt="beta formula" src="https://github.com/user-attachments/assets/a30e0a88-aa2b-4909-b69e-dfd085a7a341" />


