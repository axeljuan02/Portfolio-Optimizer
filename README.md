# SORA — The Smart Optimized Risk Allocator

**SORA** is the fastest evolving project at the crossroads of **quantitative finance**, **programming**, and **investment education**.  
It began as a deep dive into the **institutional investment world** to understand how real portfolios are built, how risk is managed, and how professional models work in practice.  

As I sought to **relearn, better understand, and go beyond** the finance concepts studied at ESSEC to apply them to my own portfolio, I faced a **massive gap** separating institutions from everyday investors. 
> Many modern platforms and “finance influencers” promote an illusion of *easy, accessible investing*, while the reality is far more complex.  

This project was born from the belief that **everyone should have access to the same elite models and tools once reserved for institutional desks**. The ambition quickly grew: to build a **professional-grade portfolio optimization engine** that is **powerful, transparent, and accessible**, turning the complex theories and tools of Wall Street into something **anyone can learn from and use**.  

SORA is no longer just an optimizer, it’s a mission to **educate as much as it optimizes**, demystifying institutional models, breaking down the gap, and empowering individuals to make **truly informed investment decisions**.  



https://github.com/user-attachments/assets/36d53164-234e-40f6-9592-dbeb52e12531


<br>


## ✅ Key Features

SORA is a multi-page Streamlit MVP that operationalizes a Python quantitative engine for portfolio construction and risk analysis. It is designed as an auditable, reproducible demonstrator suitable for academic review and as a foundation for a future SaaS.

| Page | Purpose | Key outputs |
|---|---|---|
| Portfolio Setup | Define tickers, dates, and weights; validate inputs and constraints | Cleaned universe, normalized weights, session state |
| Efficient Frontier | Compute constrained mean–variance frontier and optimal points | Frontier curve, max-Sharpe and min-volatility portfolios, allocation tables |
| Monte Carlo Risk | Simulate paths from estimated moments and correlations | VaR and CVaR from simulated P&L, loss distributions, stress scenarios |
| Factor Attribution (FF5) | Regress excess returns on Mkt–RF, SMB, HML, RMW, CMA | Alpha, betas, t-statistics, R² and adjusted R², factor loading charts |
| Explanations | Translate results into plain-English narratives | Contextual summaries, decision notes, caveats |
| Export | Persist selected figures and tables | PNG charts, CSV summaries, reproducibility notes |

**Methods and metrics :** Efficient Frontier (Markowitz), Monte Carlo simulation, and Fama–French 5 attribution with KPIs including CAGR, annualized volatility, Sharpe, Sortino, and Max Drawdown. Long-only constraints with per-asset caps, frequency-consistent risk-free handling, documented annualization, and reproducible seeds. Live data via Yahoo Finance and Fama–French, with the option to run on bundled sample datasets.


<br>


## ⚙️ Processing Pipeline

<img width="1684" height="470" alt="sora pipeline" src="https://github.com/user-attachments/assets/2f806eb6-5efd-4df5-9a5a-3f60cc32cfb8" />



<br>
<br>


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



<br>



## 📁 About This Repository

This repository contains the **SORA Core Engine**, the quantitative foundation powering the SORA platform.  

Initially, it was built simply to code financial models in Python and demonstrate to universities and professionals both my understanding of financial principles and my ability to implement them programmatically.  
It is now part of a broader project that includes a multi-page Streamlit MVP, with a SaaS-oriented WebApp in private development.

Detailed explanations of each component are provided in their dedicated documentation:
- [Educational_Showcase README](Educational_Showcase.md) — Educational focus, models, and algorithms for Admissions Committees.  
- [SORA_WebApp README](SORA_WebApp.md) — MVP architecture, features, and product roadmap for fellow Finance students and developpers.



<br>



## 👤 Founder & Vision

**Axel JUAN**  
BBA Third-Year Student @ ESSEC Business School  
CFA Level 1 Candidate  
Personal portfolio manager since age 15  

SORA is part of a broader academic and entrepreneurial journey, combining technical rigor with a long-term vision: to give investors the knowledge and tools once reserved for professionals.  

It is under active development and is designed to showcase the engine’s capabilities, coding standards, and quantitative rigor for academic admissions and professional evaluation.  


For demo access, collaboration inquiries, or strategic discussions: [juan.axel@protonmail.com]  

