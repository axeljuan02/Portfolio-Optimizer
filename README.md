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

This project was born from a belief that **everyone deserves access to elite knowledge and tools**, that the models once reserved for institutional desks could be **demystified and made usable**.  

And that’s when SORA stopped being just a coding experiment, and started its transformation into a project with a mission:  
→ **To close the gap. To democratize professional investing.**



## ⚠️ About This Repository

This public repository **SORA Core Engine**, showcases the full **Python-based foundation** of the SORA project: a modular portfolio optimizer with quantitative models like **Markowitz**, **Monte Carlo simulation**, **Fama-French 5 factors**, and a first integration of **AI/NLP logic**.  
It is made publicly available for **educational, technical, and demonstrative purposes**, particularly for academic admissions, recruiters, and fellow developers.

In parallel, a **private repository** **SORA WebApp Premium** is under development. It includes all the elements necessary to turn this engine into a full SaaS product:  
A production-ready **Django backend**, a **React frontend**, advanced **AI features**, and a **macroeconomic intelligence module**.

While this public repo demonstrates the **core engine built from scratch**, not all features are showcased here, and thus by design. Some modules remain private to preserve both a **technical and strategic edge** as part of SORA’s SaaS development.  
This repo is primarily meant to **showcase coding skills and quantitative expertise** for **academic admissions** and **professional opportunities**.  



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

- **AI/NLP Assistant Integration**  
OpenAI-powered assistant connected to the full model output.  
Generates context-aware summaries and answers in real-time, directly in the terminal.  
Includes intelligent memory, full chart/data access, and **color-coded chat interface** for clarity.  


## ⚙️ How It Works

1️⃣ **User inputs**: portfolio tickers & allocations  
2️⃣ **Market data** is fetched via Yahoo Finance  
3️⃣ **Metrics calculated**: returns, volatility, Sharpe, Beta, etc.  
4️⃣ **Optimization runs**: chosen strategy (Sharpe, min-vol, etc.)  
5️⃣ **Risk engine**: Monte Carlo simulation runs → VaR & CVaR calculated  
6️⃣ **Factor model**: Fama-French 5 regression → alpha & beta extraction  
7️⃣ **AI assistant activated**: parses all results (charts, KPIs, regressions) and answers user queries via intelligent memory  
8️⃣ **Outputs delivered**: optimized weights, full analytics, risk plots, and interactive AI reports in terminal  



## 🛤️ Roadmap

| Module | Status |
|--------|--------|
| Multi-strategy optimizer | ✅ Done |
| Efficient Frontier | ✅ Done |
| Monte Carlo VaR & CVaR | ✅ Done |
| A mission, a vision, a new design | ✅ Done |
| Fama-French model | ✅ Done |
| Codebase refactor into a modular engine | ✅ Done |
| AI/NLP integration | ✅ Done |
| MVP Website development (Streamlit + SupaBase) | 🚧 In Progress |
| MVP Release & Beta testing | Planned |
| Macroeconomics Agregator Model | Planned |
| Expansion (FastAPI, React, webdev...) | Planned |



## 📸 First Results
<img width="2336" height="1122" alt="efficient_frontier_results" src="https://github.com/user-attachments/assets/5b5b5cba-2734-4f16-b10f-bafc2627e3d4" />
<img width="2388" height="1182" alt="Monte Carlo - VaR - CVaR" src="https://github.com/user-attachments/assets/1efdcda4-9fcf-43e9-a7ba-fa3f5c2c04b0" />
<img width="2046" height="998" alt="Fama French 5" src="https://github.com/user-attachments/assets/7350a18f-4dfd-44ef-85ec-2692da288273" />
<img width="2256" height="1264" alt="sora_ai1" src="https://github.com/user-attachments/assets/4c0ad102-9b6c-4a16-918c-82d40e328cc8" />
<img width="2232" height="1304" alt="sora_ai2" src="https://github.com/user-attachments/assets/28f47a62-0985-4a44-9c55-51c00b3e77ee" />
<img width="2240" height="1522" alt="sora_ai3" src="https://github.com/user-attachments/assets/a66673a5-1752-4599-851d-b3451d6c50ad" />




## 🤝 Contributions & Feedback

SORA is under active development.  
⭐ **Star** the repo if you find it useful.  
💡 Open **issues** for suggestions or ideas.  

