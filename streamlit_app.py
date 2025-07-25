import streamlit as st
import yfinance as yf
import pandas as pd

# ----- Strategy Simulation -----
def simulate_edge(df, drift_ticks=150, tick_size=0.01, error_rate=0.05):
    drift_amount = drift_ticks * tick_size
    df['sim_profit'] = df['edge'].apply(
        lambda edge: error_rate * (drift_amount - abs(edge)) if abs(edge) <= drift_amount else 0
    )
    return df

# ----- Option Chain Fetcher -----
def fetch_chain(symbol):
    tk = yf.Ticker(symbol)
    expiries = tk.options
    rows = []

    try:
        underlying_price = tk.history(period='1d')['Close'].iloc[-1]
    except Exception:
        st.warning(f"Could not fetch price for {symbol}")
        return pd.DataFrame()

    st.markdown(f"**{symbol} Spot Price:** ${underlying_price:.2f}")

    for exp in expiries[:3]:
        st.write(f"Expiration: `{exp}`")
        try:
            chain = tk.option_chain(exp)
            calls = chain.calls.set_index('strike')
            puts = chain.puts.set_index('strike')
        except Exception:
            continue

        common_strikes = calls.index.intersection(puts.index)
        # Filter strikes within ±10% of current price for performance
        strikes = [s for s in common_strikes if 0.9 * underlying_price <= s <= 1.1 * underlying_price]

        for strike in strikes:
            call = calls.loc[strike]
            put = puts.loc[strike]

            mid_call = (call['bid'] + call['ask']) / 2
            mid_put = (put['bid'] + put['ask']) / 2
            synthetic_price = mid_call - mid_put + strike
            edge = underlying_price - synthetic_price

            # Yahoo Finance options URL for this expiry and strike
            yfinance_url = (
                f"https://finance.yahoo.com/quote/{symbol}/options?date="
                f"{pd.Timestamp(exp).strftime('%s')}&strike={strike}"
            )

            rows.append({
                'expiry': exp,
                'strike': strike,
                'mid_call': mid_call,
                'mid_put': mid_put,
                'synthetic_price': synthetic_price,
                'stock_price': underlying_price,
                'edge': edge,
                'option_link': yfinance_url
            })

    df = pd.DataFrame(rows)
    return df.sort_values(by='edge', key=abs, ascending=False)

# ----- Ticker Names Map -----
ticker_names = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "SPY": "S&P 500 ETF",
    "GLD": "Gold ETF",
    "USO": "Crude Oil ETF",
    "IEF": "7–10 Yr Treasury ETF",
    "UUP": "US Dollar Index ETF"
}

# ----- Streamlit App Layout -----
st.set_page_config(page_title="Options Expiration Edge Scanner", layout="wide")
st.title("Options Expiration Edge Scanner")
st.caption("Built by Faraz Hakim | Harvard '27, Physics and Statistics")

st.markdown("""
This tool scans for inefficiencies in synthetic options pricing near expiration. The strategy is rooted in conversion/reversal arbitrage—constructing synthetic stock positions (long call + short put) and offsetting them with the underlying.

Mispricings often emerge between 2–4 PM ET on expiration day due to execution errors: counterparties sometimes fail to cancel in-the-money options or mistakenly exercise out-of-the-money ones. These behavioral mistakes create small but statistically exploitable edges.

This scanner identifies those edges by comparing synthetic prices to spot, simulating profitability from assignment drift. It is not a trading system, but a research and signal-exploration tool built for educational purposes. I selected a few tickers from stocks, commodities, and ETFs, though any other ticker may be added in the code and scanned as well.
""")

st.info("Data is sourced from Yahoo Finance via the yfinance Python package.")

# ----- Auto-run Scan for All Tickers -----
tickers = list(ticker_names.keys())
cols_per_row = 2

# Add refresh timestamp
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

for i in range(0, len(tickers), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, symbol in enumerate(tickers[i:i+cols_per_row]):
        with cols[j]:
            readable_name = ticker_names[symbol]
            st.subheader(f"{symbol} — {readable_name}")

            df = fetch_chain(symbol)
            if df.empty:
                st.write(f"No data available for {symbol}.")
                continue

            # Hide expired options (edge case)
            df = df[pd.to_datetime(df['expiry']) >= pd.Timestamp.today()]

            df = simulate_edge(df)
            df['ticker'] = symbol

            display_df = df[['expiry', 'strike', 'edge', 'sim_profit', 'option_link']].head(10).copy()
            # Round columns for cleaner display
            display_df['edge'] = display_df['edge'].round(4)
            display_df['sim_profit'] = display_df['sim_profit'].round(4)
            # Rename columns for clarity
            display_df.rename(columns={
                'edge': 'Edge ($)',
                'sim_profit': 'Simulated Profit ($)'
            }, inplace=True)
            display_df['Yahoo Finance'] = display_df['option_link'].apply(
                lambda url: f'<a href="{url}" target="_blank">View Option</a>'
            )
            display_df = display_df[['expiry', 'strike', 'Edge ($)', 'Simulated Profit ($)', 'Yahoo Finance']]
            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# ----- Footer -----
st.markdown("---")
st.caption("For educational and demonstration purposes only. Not investment advice.")

st.markdown(
    '[View Source on GitHub](https://github.com/farazhkpineapple/options-edge-scanner)',
    unsafe_allow_html=True
)





