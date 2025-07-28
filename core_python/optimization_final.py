import yfinance as yf                               # données yahoo finance
import pandas as pd                                 # pour avoir des DataFrames (tableau)
import numpy as np                                  # pour les calculs
from datetime import datetime, timedelta
from scipy.optimize import minimize                 # pour l'optimisation et stratégies






# Toutes les Données et Inputs

tickers = input("Entrez vos actifs (ticker) séparés d'une virgule comme suit puis taper sur entrée : SPY, GLD, BND, BTC-USD, ... :\n")
tickers = [t.strip() for t in tickers.split(',')]  # Convertit la chaîne d'entrée en liste [] d'actifs


initial_weights = np.array([float(input(f"Entrez votre allocation sur 1 (20% = 0.2) pour {ticker} : ")) for ticker in tickers]) 


end_date = datetime.today()
annees = int(input("Entrez le nombre d'années que vous voulez prendre en considération (par exemple, 5 pour 5 ans) : "))
start_date = end_date - timedelta(annees * 365)


tableau_df = pd.DataFrame()

for x in tickers :                                  # pour chaque asset de la liste, download toutes ses datas yahoo finance des x dernieres années en input
    data = yf.download(x, start_date, end_date)     # Remplis la DataFrame
    tableau_df[x] = data['Close']


risk_free_rate = 0.02 / 252 

# Download des données du marché de référence (SPY) pour le calcul du beta
market_ticker = "SPY"
market_data = yf.download(market_ticker, start_date, end_date)
market_returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()









# Calculs des Formules et KPIs


log_returns = np.log(tableau_df / tableau_df.shift(1))      # calcul des rendements log ->  plus stable et symétrique que  rendement simple
log_returns = log_returns.dropna()                          # supprimer les lignes avec des valeurs NaN 

# Calcul de la matrice de covariance des rendements
cov_matrix = log_returns.cov()*252                          # matrice(tableau) de covariance des rendements log entre nos actifs, 
#print(cov_matrix)                                          # *252 pour annualiser la covariance (252 jours de bourse par an)


def standard_deviation(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))      # formule de l'écart-type des rendements pondérés par les poids des actifs grace aux méthodes numpy (np.dot) pour les multiplications matricielles ou @

def exp_returns(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252                   # exp_returns != log_returns car le deuxieme est ~ le rendement jounalier.

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    returns = exp_returns(weights, log_returns)                         # calcul des rendements espérés
    volatility = standard_deviation(weights, cov_matrix)                # calcul de la volatilité
    return (returns - risk_free_rate) / volatility                      # calcul du Sharpe ratio : stratégie visant a maximiser le ratio, Viser le 2.

def beta(weights, log_returns, market_returns):
    """
    Calcule le beta moyen du portefeuille par rapport à un indice de marché --> le SP500 'SPY' pour l'instant.

    Parameters:
    - weights: array (liste) des poids du portefeuille (somme = 1)
    - log_returns: DataFrame des rendements log des actifs du portefeuille
    - market_returns: Series des rendements log du marché (ex: SPY)
    """

    cov_with_market = log_returns.apply(lambda x: x.cov(market_returns)) * 252      # Covariance entre chaque actif et le marché (annualisée)
    var_market = market_returns.var() * 252                                         # Variance du marché (annualisée)
    
    betas = cov_with_market.values / var_market                                     # Convert betas to numpy array
    weights = np.array(weights)                                                     # Ensure weights is a numpy array
    beta_portfolio = np.dot(weights, betas)                                         # Beta moyen pondéré du portefeuille 
    
    return beta_portfolio


# Aligne les index pour éviter les bugs de covariance
common_index = log_returns.index.intersection(market_returns.index)
log_returns_aligned = log_returns.loc[common_index]
market_returns_aligned = market_returns.loc[common_index]

# Supprime les lignes avec des NaN pour toutes les colonnes et le marché
aligned_df = log_returns_aligned.copy()
aligned_df['market'] = market_returns_aligned
aligned_df = aligned_df.dropna()

log_returns_final = aligned_df.drop(columns=['market'])
market_returns_final = aligned_df['market']









# Optimisation et Choix des Stratégies


chosed_strategy = input("Quelle stratégie d'optimisation voulez-vous utiliser ?\n"
                        "1. Max Sharpe Ratio\n"
                        "2. Max Expected Returns\n"
                        "3. Min Volatility\n"
                        "Entrez 1, 2 ou 3 : ")



# Définition des contraintes et des bornes pour l'optimisation
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}        # somme des poids = 1 (100% du portefeuille investi) Complexe...
bounds = [(0, 0.5) for _ in range(len(tickers))]                                # pas de short, ni de + 50% par actif.



if chosed_strategy == '1': # Max Sharpe Ratio Strategy
    def negative_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    optimized_results = minimize(negative_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
    optimized_weights = optimized_results.x
    print("Optimisation du portefeuille pour maximiser le Sharpe Ratio...")



elif chosed_strategy == '2': # Max Expected Return Strategy
    def negative_expected_returns(weights, log_returns):
        return -exp_returns(weights, log_returns)

    optimized_results = minimize(negative_expected_returns, initial_weights, args=(log_returns,), method='SLSQP', constraints=constraints, bounds=bounds)
    optimized_weights = optimized_results.x
    print("Optimisation du portefeuille pour maximiser les rendements espérés...")



elif chosed_strategy == '3': # Min Volatility Strategy
    def negative_volatility(weights, cov_matrix):
        return standard_deviation(weights, cov_matrix)

    optimized_results = minimize(negative_volatility, initial_weights, args=(cov_matrix,), method='SLSQP', constraints=constraints, bounds=bounds)
    optimized_weights = optimized_results.x
    print("Optimisation du portefeuille pour minimiser la volatilité...")

else:
    raise ValueError("Stratégie non reconnue. Veuillez entrer 1, 2 ou 3.")               









# Affichage des Résultats et Comparaison des Portefeuilles

print("")

print("Initial Weights:")
for ticker, weight in zip(tickers, initial_weights):
    print(f"{ticker}: {weight:.4f}")

initial_exp_returns = exp_returns(initial_weights, log_returns)
print(f"Initial Expected Annual Return: {initial_exp_returns:.4f}")

initial_portfolio_volatility = standard_deviation(initial_weights, cov_matrix)
print(f"Initial Volatility: {initial_portfolio_volatility:.4f}")

initial_portfolio_beta = beta(initial_weights, log_returns_final, market_returns_final)
print(f"Initial Beta: {initial_portfolio_beta:.4f}")

initial_sharpe_ratio = sharpe_ratio(initial_weights, log_returns, cov_matrix, risk_free_rate)
print(f"Initial Sharpe Ratio : {initial_sharpe_ratio:.4f}")

print("")

print("Optimal Weights:")
for ticker, weight in zip(tickers, optimized_weights):
    print(f"{ticker}: {weight:.4f}")

optimal_portfolio_return = exp_returns(optimized_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimized_weights, cov_matrix)
optimal_portfolio_beta = beta(optimized_weights, log_returns_final, market_returns_final)
optimal_sharpe_ratio = sharpe_ratio(optimized_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Expected Beta: {optimal_portfolio_beta:.4f}")
print(f"Expected Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

print("")
