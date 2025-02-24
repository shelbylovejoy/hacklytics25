# %%
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

years = 3
selected_stock = 'AAPL'

class StockoSaurus:
    def __init__(self, ticker, years=years):
        self.client = pymongo.MongoClient("mongodb+srv://jmsherrier:G0IFFnPQmi3mGW2Z@cluster0.ij772.mongodb.net/")
        self.db = self.client["stocks"]
        self.collection = self.db["sp500"]
        self.ticker = ticker
        self.years = years
        
        doc = self.collection.find_one({"ticker": ticker})
        self.history = pd.DataFrame(doc['history'])
        self.history['date'] = pd.to_datetime(self.history['date'])
        self.history.set_index('date', inplace=True)
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        self.data = self.history[self.history.index >= cutoff]['close']

class ReturnRex(StockoSaurus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cagr = None
        self._sma = None
        self._ema = None
        self._beta = None

    def beta(self):
        if self._beta is None:
            market_data = self.collection.find_one({"ticker": "SPY"})  # S&P 500 index
            market_returns = pd.DataFrame(market_data['history'])
            market_returns['date'] = pd.to_datetime(market_returns['date'])
            market_returns.set_index('date', inplace=True)
            market_returns = market_returns['close'].pct_change().dropna()

            stock_returns = self.data.pct_change().dropna()

            # Align dates
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            
            covariance = aligned_data.cov().iloc[0, 1]
            market_variance = aligned_data.iloc[:, 1].var()
            
            self._beta = covariance / market_variance

        return self._beta
    
    def cagr(self):
        if self._cagr is None:
            start = self.data.iloc[0]
            end = self.data.iloc[-1]
            self._cagr = (end / start) ** (1/self.years) - 1
        return self._cagr

    def sma(self):
        if self._sma is None:
            self._sma = pd.DataFrame({
                'SMA_50': self.data.rolling(50).mean(),
                'SMA_200': self.data.rolling(200).mean()
            })
        return self._sma

    def ema(self):
        if self._ema is None:
            self._ema = pd.DataFrame({
                'EMA_50': self.data.ewm(span=50).mean(),
                'EMA_200': self.data.ewm(span=200).mean()
            })
        return self._ema


class Trendadactyl(StockoSaurus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._linreg = None
        self._rsi = None
        self._macd = None

    def linear_regression(self, window=30):
        if self._linreg is None:
            self._linreg = self.data.rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
            )
        return self._linreg

    def macd(self, short_period=12, long_period=26, signal_period=9):
        if self._macd is None:
            short_ema = self.data.ewm(span=short_period, adjust=False).mean()
            long_ema = self.data.ewm(span=long_period, adjust=False).mean()
            macd_line = short_ema - long_ema
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            self._macd = pd.DataFrame({
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': histogram
            })
        return self._macd

    def rsi(self, window=14):
        if self._rsi is None:
            delta = self.data.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss
            self._rsi = 100 - (100 / (1 + rs))
        return self._rsi

class Volatiliraptor(StockoSaurus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._volatility = None
        self._drawdown = None
        self._var = None
        self._sharpe = None

    def sharpe_ratio(self, risk_free_rate=0.02):
        if self._sharpe is None:
            returns = self.data.pct_change().dropna()
            excess_returns = returns - risk_free_rate / 252  # Assuming daily data
            self._sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return self._sharpe

    def historical_volatility(self, window=21):
        if self._volatility is None:
            returns = self.data.pct_change().dropna()
            self._volatility = returns.rolling(window).std() * np.sqrt(252)
        return self._volatility

    def max_drawdown(self):
        if self._drawdown is None:
            rolling_max = self.data.cummax()
            self._drawdown = (self.data - rolling_max) / rolling_max
        return self._drawdown

    def value_at_risk(self, confidence=0.95):
        if self._var is None:
            returns = self.data.pct_change().dropna()
            self._var = returns.quantile(1 - confidence)
        return self._var
    
class DinoAnalyzer:
    def __init__(self, ticker, years=years):
        self.ticker = ticker
        self.years = years
        self.return_rex = ReturnRex(ticker, years)
        self.trend_dino = Trendadactyl(ticker, years)
        self.vol_raptor = Volatiliraptor(ticker, years)

    def normalize(self, value, min_val, max_val):
        return max(0, min(1, (value - min_val) / (max_val - min_val)))

    def return_rating(self):
        cagr = self.return_rex.cagr()
        sma_ratio = self.return_rex.sma()['SMA_50'].iloc[-1] / self.return_rex.sma()['SMA_200'].iloc[-1]
        ema_ratio = self.return_rex.ema()['EMA_50'].iloc[-1] / self.return_rex.ema()['EMA_200'].iloc[-1]
        beta = self.return_rex.beta()

        cagr_score = self.normalize(cagr, -0.1, 0.3)  # Assuming -10% to 30% CAGR range
        sma_score = self.normalize(sma_ratio, 0.8, 1.2)  # Assuming 0.8 to 1.2 ratio range
        ema_score = self.normalize(ema_ratio, 0.8, 1.2)  # Assuming 0.8 to 1.2 ratio range
        beta_score = self.normalize(beta, 0, 2)  # Assuming 0 to 2 beta range

        return (cagr_score * 0.4 + sma_score * 0.2 + ema_score * 0.2 + beta_score * 0.2) * 100

    def trend_rating(self):
        slope = self.trend_dino.linear_regression().iloc[-1]
        rsi = self.trend_dino.rsi().iloc[-1]
        macd = self.trend_dino.macd()

        slope_score = self.normalize(slope, -1, 1)  # Assuming -1 to 1 slope range
        rsi_score = 1 - abs(rsi - 50) / 50  # RSI closer to 50 is better
        
        # MACD score: positive if MACD is above signal line, negative otherwise
        macd_score = self.normalize(macd['MACD'].iloc[-1] - macd['Signal'].iloc[-1], -2, 2)

        return (slope_score * 0.4 + rsi_score * 0.3 + macd_score * 0.3) * 100

    def risk_rating(self):
        volatility = self.vol_raptor.historical_volatility().iloc[-1]
        max_drawdown = abs(self.vol_raptor.max_drawdown().min())
        var = abs(self.vol_raptor.value_at_risk())
        sharpe = self.vol_raptor.sharpe_ratio()

        vol_score = 1 - self.normalize(volatility, 0, 0.5)  # Assuming 0% to 50% volatility range
        drawdown_score = 1 - self.normalize(max_drawdown, 0, 0.5)  # Assuming 0% to 50% drawdown range
        var_score = 1 - self.normalize(var, 0, 0.1)  # Assuming 0% to 10% VaR range
        sharpe_score = self.normalize(sharpe, -1, 3)  # Assuming -1 to 3 Sharpe ratio range

        return (vol_score * 0.3 + drawdown_score * 0.3 + var_score * 0.2 + sharpe_score * 0.2) * 100

    def overall_rating(self):
        return_score = self.return_rating()
        trend_score = self.trend_rating()
        risk_score = self.risk_rating()
        return (return_score * 0.4 + trend_score * 0.3 + risk_score * 0.3)

    def generate_report(self):
        return {
            'ticker': self.ticker,
            'period': f'{self.years} years',
            'return_rating': self.return_rating(),
            'trend_rating': self.trend_rating(),
            'risk_rating': self.risk_rating(),
            'overall_rating': self.overall_rating(),
            'cagr': self.return_rex.cagr(),
            'beta': self.return_rex.beta(),
            'current_price': self.return_rex.data.iloc[-1],
            'sma_50': self.return_rex.sma()['SMA_50'].iloc[-1],
            'sma_200': self.return_rex.sma()['SMA_200'].iloc[-1],
            'ema_50': self.return_rex.ema()['EMA_50'].iloc[-1],
            'ema_200': self.return_rex.ema()['EMA_200'].iloc[-1],
            'regression_slope': self.trend_dino.linear_regression().iloc[-1],
            'rsi': self.trend_dino.rsi().iloc[-1],
            'macd': self.trend_dino.macd()['MACD'].iloc[-1],
            'macd_signal': self.trend_dino.macd()['Signal'].iloc[-1],
            'macd_histogram': self.trend_dino.macd()['Histogram'].iloc[-1],
            'sharpe_ratio': self.vol_raptor.sharpe_ratio(),
            'volatility': self.vol_raptor.historical_volatility().iloc[-1],
            'max_drawdown': self.vol_raptor.max_drawdown().min(),
            'value_at_risk': self.vol_raptor.value_at_risk()
        }

# %%
analyzer = DinoAnalyzer(selected_stock, years=years)

# Generate the full report
report = analyzer.generate_report()

def get_interpretations(report):
    interpretations = {}

    # Return Rating Interpretation
    if report['return_rating'] > 80:
        interpretations['return'] = f"Excellent return potential (Rating: {report['return_rating']:.2f}%)"
    elif report['return_rating'] > 60:
        interpretations['return'] = f"Good return potential (Rating: {report['return_rating']:.2f}%)"
    elif report['return_rating'] > 40:
        interpretations['return'] = f"Average return potential (Rating: {report['return_rating']:.2f}%)"
    else:
        interpretations['return'] = f"Below average return potential (Rating: {report['return_rating']:.2f}%)"

    # Trend Rating Interpretation
    if report['trend_rating'] > 80:
        interpretations['trend'] = f"Strong upward trend (Rating: {report['trend_rating']:.2f}%)"
    elif report['trend_rating'] > 60:
        interpretations['trend'] = f"Moderate upward trend (Rating: {report['trend_rating']:.2f}%)"
    elif report['trend_rating'] > 40:
        interpretations['trend'] = f"Neutral trend (Rating: {report['trend_rating']:.2f}%)"
    else:
        interpretations['trend'] = f"Downward trend (Rating: {report['trend_rating']:.2f}%)"

    # Risk Rating Interpretation
    if report['risk_rating'] > 80:
        interpretations['risk'] = f"Low risk (Rating: {report['risk_rating']:.2f}%)"
    elif report['risk_rating'] > 60:
        interpretations['risk'] = f"Moderate risk (Rating: {report['risk_rating']:.2f}%)"
    elif report['risk_rating'] > 40:
        interpretations['risk'] = f"High risk (Rating: {report['risk_rating']:.2f}%)"
    else:
        interpretations['risk'] = f"Very high risk (Rating: {report['risk_rating']:.2f}%)"

    # Overall Rating Interpretation
    if report['overall_rating'] > 80:
        interpretations['overall'] = f"Strong Buy (Rating: {report['overall_rating']:.2f}%)"
    elif report['overall_rating'] > 60:
        interpretations['overall'] = f"Buy (Rating: {report['overall_rating']:.2f}%)"
    elif report['overall_rating'] > 40:
        interpretations['overall'] = f"Hold (Rating: {report['overall_rating']:.2f}%)"
    elif report['overall_rating'] > 20:
        interpretations['overall'] = f"Sell (Rating: {report['overall_rating']:.2f}%)"
    else:
        interpretations['overall'] = f"Strong Sell (Rating: {report['overall_rating']:.2f}%)"

    # CAGR Interpretation
    if report['cagr'] > 0.20:
        interpretations['cagr'] = f"Exceptional growth (CAGR: {report['cagr']:.2%})"
    elif report['cagr'] > 0.10:
        interpretations['cagr'] = f"Strong growth (CAGR: {report['cagr']:.2%})"
    elif report['cagr'] > 0.05:
        interpretations['cagr'] = f"Moderate growth (CAGR: {report['cagr']:.2%})"
    elif report['cagr'] > 0:
        interpretations['cagr'] = f"Slow growth (CAGR: {report['cagr']:.2%})"
    else:
        interpretations['cagr'] = f"Negative growth (CAGR: {report['cagr']:.2%})"

    # Beta Interpretation
    if report['beta'] > 1.5:
        interpretations['beta'] = f"Highly volatile compared to market (Beta: {report['beta']:.2f})"
    elif report['beta'] > 1:
        interpretations['beta'] = f"More volatile than market (Beta: {report['beta']:.2f})"
    elif report['beta'] > 0.5:
        interpretations['beta'] = f"Less volatile than market (Beta: {report['beta']:.2f})"
    else:
        interpretations['beta'] = f"Much less volatile than market (Beta: {report['beta']:.2f})"

    # RSI Interpretation
    if report['rsi'] > 70:
        interpretations['rsi'] = f"Overbought (RSI: {report['rsi']:.2f})"
    elif report['rsi'] < 30:
        interpretations['rsi'] = f"Oversold (RSI: {report['rsi']:.2f})"
    else:
        interpretations['rsi'] = f"Neutral (RSI: {report['rsi']:.2f})"

    # Sharpe Ratio Interpretation
    if report['sharpe_ratio'] > 1:
        interpretations['sharpe'] = f"Good risk-adjusted returns (Sharpe: {report['sharpe_ratio']:.2f})"
    elif report['sharpe_ratio'] > 0:
        interpretations['sharpe'] = f"Positive risk-adjusted returns (Sharpe: {report['sharpe_ratio']:.2f})"
    else:
        interpretations['sharpe'] = f"Poor risk-adjusted returns (Sharpe: {report['sharpe_ratio']:.2f})"

    # Volatility Interpretation
    if report['volatility'] > 0.4:
        interpretations['volatility'] = f"Extremely high volatility (Volatility: {report['volatility']:.2%})"
    elif report['volatility'] > 0.2:
        interpretations['volatility'] = f"High volatility (Volatility: {report['volatility']:.2%})"
    elif report['volatility'] > 0.1:
        interpretations['volatility'] = f"Moderate volatility (Volatility: {report['volatility']:.2%})"
    else:
        interpretations['volatility'] = f"Low volatility (Volatility: {report['volatility']:.2%})"

    return interpretations


# %%
def plot_stockosaurus(ticker, years=5):
    stock = StockoSaurus(ticker, years)
    
    plt.figure(figsize=(12, 6))
    plt.plot(stock.data.index, stock.data, label=f"{ticker} Closing Price", color='#b6c59d9e')
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.title(f"{ticker}")
    plt.legend()
    plt.grid()
    plt.savefig('StockoSaurus.png')
    return ('StockoSaurus.png')

def plot_returnrex1(ticker, years=5):
    stock_return = ReturnRex(ticker, years)
    sma_data = stock_return.sma()

    plt.figure(figsize=(12, 6))
    plt.plot(stock_return.data.index, stock_return.data, label=f"{ticker} Closing Price", color='#b6c59d9e')
    plt.plot(sma_data.index, sma_data["SMA_50"], label=f"{ticker} 50 Day Simple Moving Average", color="green")
    plt.plot(sma_data.index, sma_data["SMA_200"], label=f"{ticker} 200 Day Simple Moving Average", color="orange")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.title(f"{ticker} Stock Growth and Simple Moving Averages")
    plt.legend()
    plt.grid()
    plt.savefig('ReturnRex1.png')
    return ('ReturnRex1.png')

def plot_returnrex2(ticker, years=years):
    stock_return = ReturnRex(ticker, years)
    ema_data = stock_return.ema()
    
    plt.figure(figsize=(12, 6))
    plt.plot(stock_return.data.index, stock_return.data, label=f"{ticker} Closing Price", color='#b6c59d9e')
    plt.plot(ema_data.index, ema_data["EMA_50"], label=f"{ticker} 50 Day Exponential Moving Average", color="green")
    plt.plot(ema_data.index, ema_data["EMA_200"], label=f"{ticker} 200 Day Exponential Moving Average", color="orange") 
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.title(f"{ticker} Stock Growth and Exponential Moving Averages")
    plt.legend()
    plt.grid()
    plt.savefig('ReturnRex2.png')
    return ('ReturnRex2.png')

def plot_volitiliraptor(ticker, years=years):
    stock_volatility = Volatiliraptor(ticker, years)
    volatility = stock_volatility.historical_volatility()

    plt.figure(figsize=(12, 6))
    plt.plot(volatility.index, volatility*100, label=f"{ticker} Historical Volatility", color='#b6c59d9e')  
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    plt.title(f"{ticker} Risk Overview")
    plt.legend()
    plt.grid()
    plt.savefig('Volitiliraptor.png')
    return ('Volitiliraptor.png')

def plot_trendadactyl(ticker, years=years):
    stock_trend = Trendadactyl(ticker, years)
    
    # Prepare data for linear regression
    x = np.arange(len(stock_trend.data))
    y = stock_trend.data.values
    
    # Calculate linear regression
    n = len(x)
    m = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    b = (np.sum(y) - m * np.sum(x)) / n
    
    # Generate prediction line
    regression_line = m * x + b

    plt.figure(figsize=(12, 6))
    plt.plot(stock_trend.data.index, stock_trend.data, label=f"{ticker} Closing Price", color='#b6c59d9e')
    
    # Add Linear Regression line
    plt.plot(stock_trend.data.index, regression_line, label="Linear Regression", color='orange', linestyle="--")
    
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.title(f"{ticker} Trend and Forecasting")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
stock_name = "MSFT"
plot_trendadactyl(stock_name)
plot_stockosaurus(stock_name)
plot_returnrex1(stock_name)
plot_returnrex2(stock_name)
plot_volitiliraptor(stock_name)