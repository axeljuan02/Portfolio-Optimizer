# SORA — The Smart Optimized Risk Allocator

**SORA** is the fastest evolving project at the crossroads of **quantitative finance**, **programming**, and **investment education**.  

It began as a deep dive into the **world of institutional investing**, a way to understand how real portfolios are built, how risk is managed, and how professional models work in practice.  

Its ambition quickly grew: to build a **professional-grade portfolio optimization engine** that is **powerful, transparent, and accessible**, turning the complex theories and tools of Wall Street into something **anyone** can learn from and use.  

> *“Democratizing intelligent and professional investing for everyone, because no, investment isn't that easy.”*

SORA isn’t just an optimizer. It’s a mission to **educate as much as it optimizes**, revealing the models used by institutions, breaking down the knowledge gap, and empowering individual investors to make **truly informed decisions**.



## 📌 A Mission

When I began this project, my first goal was simple: to truly understand how **institutional and professional investing** works, the **real investment world** beyond what’s marketed to individuals.  

I wanted to revisit and apply the **finance concepts I studied at ESSEC**, not just to test them, but to **professionalize my own investments**.  

As I explored deeper, diving into the theories and models used by today’s financial giants, I saw the **massive knowledge gap** between institutions and everyday investors. It became clear how many modern investment platforms and “finance influencers” were selling the illusion of *simple, accessible investing for everyone*.  

But the truth is: it’s not that simple.  

This project was born from a belief that **everyone deserves access to elite knowledge and tools** — that the models once reserved for institutional desks could be **demystified and made usable**.  

And that’s when SORA stopped being just a coding experiment, and started its transformation into a project with a mission:  
➡️ **To close the gap. To democratize professional investing.**



## ✅ Current Features

- **Multi-Strategy Optimizer**  
3 core strategies: **Max Sharpe ratio**, **Max Expected Return**, **Min Volatility**  
KPIs: Sharpe, Beta, Volatility, Expected Return  
Portfolio constraints: no short-selling, capped allocation per asset  

- **Harry Markowitz's Efficient Frontier Generator**  
Simulation of thousands of portfolios   
Visualization of the **risk/return trade-off**    
Highlighting the optimal portfolio allocations vs the actual portfolio allocations

- **Monte Carlo Risk Engine**  
**VaR** (Value at Risk) and **CVaR** (Conditional VaR) computed through **Monte Carlo simulation**  
Loss distribution plots and extreme risk scenarios  

- **Fama-French 5 Factor Model Integration**  
Regression of the portfolio’s **excess returns** on the five Fama-French factors  
Extraction of **alpha** (manager skill) and **betas** (factor exposures)  
Bar chart visualization of factor loadings with **R²** and **Adjusted R²** annotations  

## ⚙️ How It Works

1️⃣ **User inputs**: portfolio tickers & allocations  
2️⃣ **Market data** is fetched via Yahoo Finance  
3️⃣ **Metrics calculated**: returns, volatility, Sharpe, Beta, etc.  
4️⃣ **Optimization runs**: chosen strategy (Sharpe, min-vol, etc.)  
5️⃣ **Risk engine**: Monte Carlo simulation runs → VaR & CVaR calculated  
6️⃣ **Outputs delivered**: optimized weights, KPIs, efficient frontier & risk visualizations



## 🛤️ Roadmap

| Module | Status |
|--------|--------|
| Multi-strategy optimizer | ✅ Done |
| Efficient Frontier | ✅ Done |
| Monte Carlo VaR & CVaR | ✅ Done |
| A mission, a vision, a new design | ✅ Done |
| Fama-French model | ✅ Done |
| Codebase refactor into a modular engine | ✅ Done |
| AI/NLP integration | 🚧 In Progress |
| Macroeconomics Agregator Model | Planned |
| Streamlit MVP (web app) | Planned |
| SaaS development (Django + Frontend) | Planned |
| Further Optimization | Planned |
| First release | Planned |



## 📸 First Results
<img width="2336" height="1122" alt="efficient_frontier_results" src="https://github.com/user-attachments/assets/5b5b5cba-2734-4f16-b10f-bafc2627e3d4" />
<img width="2388" height="1182" alt="Monte Carlo - VaR - CVaR" src="https://github.com/user-attachments/assets/1efdcda4-9fcf-43e9-a7ba-fa3f5c2c04b0" />
<img width="2046" height="998" alt="Fama French 5" src="https://github.com/user-attachments/assets/7350a18f-4dfd-44ef-85ec-2692da288273" />



## 🤝 Contributions & Feedback

SORA is under active development.  
⭐ **Star** the repo if you find it useful.  
💡 Open **issues** for suggestions or ideas.  

