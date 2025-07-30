from datetime import datetime, timedelta
import pandas_datareader.data as web
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt



end_date = datetime.today()
years = int(input('Enter the number of years you want to take into consideration (e.g : 5) : '))
start_date = end_date - timedelta(years*365)

tickers = input('Enter the assets you own by their tickers and separated by a comma (e.g : AMZN, AAPL, NVDA, GOOGL) : ')
ticker_list = tickers.split(sep=',')
ticker_list = [t.strip() for t in ticker_list]

weights = input('Enter your allocation for each asset separated by a comma (e.g : 0.25,0.25,0.25,0.25) : ')
weights_list = weights.split(',')
weights_int = list(map(float, weights_list))

check = 0
for weight in weights_int:
    check += weight

if check != 1:
    print("Your allocation isn't equal to 1, please retype it :")



ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start= start_date, end= end_date)

monthly_factors = ff_data[0]        # données mensuelles
monthly_factors.head()              # bonnes datas, verifiées sur le csv téléchargé
updated_factors = monthly_factors.div(100)      # Enlève le pourcentage en divisant par 100 les data du DataFrame pandas
updated_factors.index = updated_factors.index.to_timestamp().to_period('M')



tickers_data = yf.download(ticker_list, start_date, end_date)
# Calcul des rendements mensuels desormais pour fit avec les data mensuels de Fama & French
monthly_tickers = tickers_data.resample('M').last()
monthly_returns = monthly_tickers.pct_change()
monthly_returns['Close']

# Calcul du portfolio returns en faisant la somme des rendements mensuels des actifs multipliés par leur pondération
portfolio_returns = (monthly_returns['Close'] * weights_int).sum(axis=1)
print(portfolio_returns)        # rendement du portefeuille par mois
portfolio_avg_return = portfolio_returns.mean()
print(f"Rendement moyen du portefeuille sur toute l'étendue de la date choisie : {portfolio_avg_return*100:.2f} %")     # rendement moyen du portefeuille sur toute l'étendue de la date choisie



# Aligner les données sur les memes index pour pouvoir faire la régression plus tard
portfolio_returns.index = portfolio_returns.index.to_period('M')
common_index = portfolio_returns.index.intersection(updated_factors.index)
portfolio_returns = portfolio_returns.loc[common_index]
updated_factors = updated_factors.loc[common_index]

# Garde uniquement les dates communes
common_idx = portfolio_returns.index.intersection(updated_factors.index)
portfolio_returns = portfolio_returns.loc[common_idx]
updated_factors = updated_factors.loc[common_idx]



# Calcul du rendement excédentaire
portfolio_exreturns = portfolio_returns - monthly_factors['RF'].loc[common_idx]

# Aligner X et Y
factors = updated_factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
X = sm.add_constant(factors)
Y = portfolio_exreturns

reg = sm.OLS(Y,X)

results = reg.fit()
results.params
print(results.summary())



# Derniers préparatifs avant de plot !
alpha = results.params['const']  
betas = results.params.drop('const')                        # Garde seulement les facteurs soit les betas
final_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
betas = betas[final_factors]                                # filtre seulement ces facteurs



# Plotting
betas.plot(kind='bar', figsize=(12,6),)
plt.ylabel('Betas coefficient')
plt.title('Portfolio Return Explanation – Fama-French 5 Factor Model')

# Ajout du R-squared et Adjusted R-squared sur le graphique
r_squared = results.rsquared * 100                      # R² en %
r_squared_adj = results.rsquared_adj * 100              # R² ajusté en %
x_pos = plt.xlim()[1] * 1.01   
y_pos = plt.ylim()[1] * 0.9   
plt.text(x=x_pos, y=y_pos,
         s=f"R² = {r_squared:.2f}%\nAdj. R² = {r_squared_adj:.2f}%",
         fontsize=12,
         ha='left', va='top',
         bbox=dict(facecolor="white", alpha=0.7))
plt.subplots_adjust(left=0.095, right=0.86)

plt.show()
