import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  datetime import datetime, timedelta


tickers = input("Entrez vos actifs (ticker) séparés d'une virgule comme suit puis taper sur entrée : SPY, GLD, BND, BTC-USD, ... :\n")
tickers = [t.strip() for t in tickers.split(',')]  # Convertit la chaîne d'entrée en liste [] d'actifs

end = datetime.today()
annees = int(input("Entrez le nombre d'années que vous voulez prendre en considération (ex : 5) : "))
start = end - timedelta(annees * 365)

stock_data = yf.download(tickers, start, end)
returns = stock_data['Close'].pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()



weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)

returns['portfolio'] = returns @ weights                # ajout de la colomne portfolio (retours pondérés de chaque actifs)

expected_returns = np.sum(mean_returns * weights) * 252



# PARTIE SIMULATION
time = 252
nb_sim = int(input('Entrez le nombre de simulations souhaitées (1000 à 100000) :'))

mean_returns_matrix = np.full(shape=(time, len(weights)), fill_value=mean_returns).T
portfolio_simulation = np.zeros(shape=(time, nb_sim))               # deuxieme matrice vide pleines de 0

for i in range(0, nb_sim):
    correlatedd_random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time)
    portfolio_simulation[:, i] = np.cumprod(np.inner(weights, correlatedd_random_returns) + 1) * 1000




# PARTIE CALCULS
def mcVaR(returns,alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns,alpha)
    else:
        return TypeError('Expected a pandas data series')


def mcCVaR(returns,alpha=5):
    if isinstance(returns, pd.Series):
        below_var = returns <= mcVaR(returns, alpha=alpha)
        return returns[below_var].mean()
    else:
        return TypeError('Expected a pandas data series')


portfolio_results = pd.Series(portfolio_simulation[-1,:])

#portfolio_value = 1000
var = (1000 - mcVaR(portfolio_results, alpha=5)) / 1000 * 100
cvar = (1000 - mcCVaR(portfolio_results, alpha=5)) / 1000 * 100




# PARTIE GRAPHIQUE
plt.figure(figsize=(12, 6))
plt.plot(portfolio_simulation)
plt.title('Simulation de Monte Carlo')
plt.xlabel('Durée sur une année de trading (252j)')
plt.ylabel('Valeur du portefeuille en $')

# Ajout de la VaR et CVaR sur le graphique
plt.text(x=len(portfolio_simulation)*1.1, y=portfolio_simulation.max()*0.9,
         s=f"VaR: {var:.2f}%\nCVaR: {cvar:.2f}%\nSimulation via \n{nb_sim} itérations",
         fontsize=12, bbox=dict(facecolor="white", alpha=0.7))
plt.tight_layout()

# Ajout de la ligne horizontale pour la VaR (5e percentile final)
plt.axhline(y=mcVaR(portfolio_results, alpha=5), color='black', linewidth=1, label='VaR (Confidence level : 5%)')
plt.legend()


plt.show()
