# SORA Core Engine for Educational Purpose


This public repository **SORA Core Engine**, showcases the full **Python-based foundation** of the SORA project: a modular portfolio optimizer with quantitative models like **Markowitz**, **Monte Carlo simulation**, **Fama-French 5 factors**, and a first integration of **AI/NLP logic**.  
It is made publicly available for **educational, technical, and demonstrative purposes**, particularly for academic admissions, recruiters, and fellow developers.


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



## 📸 Terminal Based Results
<img width="2336" height="1122" alt="efficient_frontier_results" src="https://github.com/user-attachments/assets/5b5b5cba-2734-4f16-b10f-bafc2627e3d4" />
<img width="2388" height="1182" alt="Monte Carlo - VaR - CVaR" src="https://github.com/user-attachments/assets/1efdcda4-9fcf-43e9-a7ba-fa3f5c2c04b0" />
<img width="2046" height="998" alt="Fama French 5" src="https://github.com/user-attachments/assets/7350a18f-4dfd-44ef-85ec-2692da288273" />
<img width="2256" height="1264" alt="sora_ai1" src="https://github.com/user-attachments/assets/4c0ad102-9b6c-4a16-918c-82d40e328cc8" />
<img width="2232" height="1304" alt="sora_ai2" src="https://github.com/user-attachments/assets/28f47a62-0985-4a44-9c55-51c00b3e77ee" />
<img width="2240" height="1522" alt="sora_ai3" src="https://github.com/user-attachments/assets/a66673a5-1752-4599-851d-b3451d6c50ad" />

