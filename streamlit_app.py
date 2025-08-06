import streamlit as st
import yfinance as yf
import pandas as pd
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----- Logger for Mispricings -----
def log_mispricings(df, symbol, threshold=0.05, log_path="mispricing_log.csv"):
    df_filtered = df[abs(df['edge']) >= threshold].copy()
    if df_filtered.empty:
        return

    now = pd.Timestamp.utcnow()
    df_filtered['logged_at'] = now
    df_filtered['symbol'] = symbol

    cols_to_log = [
        'logged_at', 'symbol', 'expiry', 'strike',
        'mid_call', 'mid_put', 'synthetic_price',
        'stock_price', 'edge'
    ]

    # Load existing log if present
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df['logged_at'] = pd.to_datetime(log_df['logged_at'], errors='coerce')
        log_df = log_df[log_df['logged_at'].notna()]

        # Keep logs within last 7 days
        cutoff = now - pd.Timedelta(days=7)
        log_df = log_df[log_df['logged_at'] >= cutoff]

        # For each row in df_filtered, keep only if not seen in last 60 minutes
        merged = pd.merge(
            df_filtered,
            log_df[['symbol', 'expiry', 'strike', 'logged_at']],
            on=['symbol', 'expiry', 'strike'],
            how='left',
            suffixes=('', '_prev')
        )
        merged['logged_at_prev'] = pd.to_datetime(merged['logged_at_prev'], errors='coerce')
        merged['time_since_last_log'] = (now - merged['logged_at_prev']).dt.total_seconds()

        # Only keep rows not seen or logged more than 1 hour ago
        to_log = merged[(merged['logged_at_prev'].isna()) | (merged['time_since_last_log'] > 3600)]
        df_filtered = to_log[cols_to_log]
    else:
        df_filtered = df_filtered[cols_to_log]

    if not df_filtered.empty:
        df_filtered.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

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

    # Display spot price and expiries compactly on one line
    expiry_str = " | ".join(expiries[:3])
    st.markdown(f"**{symbol} Spot Price:** ${underlying_price:.2f} | **Expiries:** {expiry_str}")

    for exp in expiries[:3]:
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
            exp_timestamp = int(pd.Timestamp(exp).timestamp())
            yfinance_url = f"https://finance.yahoo.com/quote/{symbol}/options?date={exp_timestamp}"

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
    if df.empty:
        return df
    return df.sort_values(by='edge', key=abs, ascending=False)

# ----- Ticker Names Map -----
ticker_names = {
    "GLD": "Gold ETF (proxy for Gold Futures)",
    "USO": "Crude Oil ETF (proxy for Crude Futures)",
    "TLT": "20+ Yr Treasury Bond ETF (proxy for ZB Futures)",
    "IEF": "7–10 Yr Treasury ETF (proxy for ZN Futures)",
    "UNG": "Natural Gas ETF (proxy for NG Futures)",
    "SLV": "Silver ETF (proxy for Silver Futures)",
    "VIXY": "Short-Term VIX Futures ETF (proxy for VX Futures)"
}

# ----- Streamlit App Layout -----
st.set_page_config(page_title="Options Expiration Edge Scanner", layout="wide")

# CSS to prevent page jumping during loading
st.markdown("""
<style>
    /* Ensure smooth loading and prevent layout shifts */
    .block-container {
        min-height: 100vh;
    }
    
    /* Stabilize layout during content loading */
    [data-testid="stDataFrame"] {
        min-height: 350px;
    }
    
    /* Smooth transitions */
    * {
        transition: opacity 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ----- Prevent Layout Jumps -----
# CSS styles above ensure stable layout during content loading
    
# ----- Sidebar Navigation -----
st.sidebar.title("Navigation")

st.sidebar.markdown("**Jump to Section:**")

# Create navigation links
st.sidebar.markdown("📊 [Summary Table & Analytics](#summary-table-by-ticker-past-7-days)")
st.sidebar.markdown("---")
st.sidebar.markdown("**Ticker Sections:**")

for symbol in ticker_names.keys():
    readable_name = ticker_names[symbol].split(" (")[0]  # Get short name
    st.sidebar.markdown(f"📈 [{symbol} - {readable_name}](#{symbol.lower()}-{readable_name.lower().replace(' ', '-')})")

st.title("Options Mispricings and Edge Logger")
st.caption("Built by Faraz Hakim | Harvard '27, Physics and Statistics")

st.markdown("""
I built this tool to explore, scan, and log inefficiencies in options pricing, and gain insights into options market behavior.

Specifically, I based this tool on the idea of **conversion/reversal arbitrage**—constructing synthetic stock positions (long call + short put) and offsetting them with the underlying.

The scanner compares the synthetic spot price (Call - Put + Strike) to the actual underlying price and highlights cases where the difference ("edge") may indicate potential mispricing.

Every time the scanner is run, potential mispricings over $0.10 are logged to a file. This models how a trade desk might collect signals to later review and analyze.
""")

st.markdown("<small style='color: gray;'>*Data is sourced from Yahoo Finance via the yfinance Python package.</small>", unsafe_allow_html=True)

# ----- Helper Functions -----
def load_and_filter_log(log_path="mispricing_log.csv", days=7):
    """Load and filter log data for the past N days"""
    if not os.path.exists(log_path):
        return pd.DataFrame()
    
    log_df = pd.read_csv(log_path)
    log_df['logged_at'] = pd.to_datetime(log_df['logged_at'], errors='coerce')
    log_df = log_df[log_df['logged_at'].notna()]
    
    # Normalize timezone
    if log_df['logged_at'].dt.tz is not None:
        log_df['logged_at'] = log_df['logged_at'].dt.tz_localize(None)
    
    # Filter for recent data
    cutoff = pd.Timestamp.utcnow().replace(tzinfo=None) - timedelta(days=days)
    return log_df[log_df['logged_at'] >= cutoff]

# ----- Pattern Recognition & Statistical Insights -----

# --- Pattern Recognition & Statistical Insights ---
def analyze_patterns(log_df):
    if log_df.empty:
        st.info("No log data available for pattern analysis.")
        return

    # Preprocessing
    log_df['logged_at'] = pd.to_datetime(log_df['logged_at'])
    log_df['day_of_week'] = log_df['logged_at'].dt.day_name()
    log_df['hour'] = log_df['logged_at'].dt.hour

    st.markdown("**Edge Distribution by Ticker**")
    st.markdown("The following chart shows how mispricings (edge) are spread for each ticker this week.")
    fig, ax = plt.subplots(figsize=(10, 6))  # Set fixed size
    sns.boxplot(data=log_df, x='symbol', y='edge', ax=ax)
    ax.set_title("Edge ($) by Ticker")
    st.pyplot(fig)

    st.markdown("""
Taller box = more inconsistent mispricings  
Box above zero = synthetic often undervalued  
Box below zero = synthetic often overvalued  
Flat box = more efficient pricing  
Use this to spot which tickers have frequent or big mispricings.
""")

    # ECDF Plot of Edge Sizes
    st.markdown("**ECDF of Mispricing Edge Magnitude**")
    st.markdown("This shows the cumulative distribution of absolute edge values by ticker.")
    log_df['abs_edge'] = log_df['edge'].abs()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for symbol in log_df['symbol'].unique():
        data = log_df[log_df['symbol'] == symbol]['abs_edge'].sort_values()
        y = np.arange(1, len(data)+1) / len(data)
        ax2.plot(data, y, marker='.', linestyle='none', label=symbol, markersize=4)
    ax2.set_xlabel("Absolute Edge ($)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend()
    ax2.set_title("Empirical CDF of Mispricing Edge")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    st.markdown("""
**Reading the ECDF:**  
• Each line shows one ticker's edge magnitude distribution  
• Steeper curves = more consistent edge sizes  
• Lines further right = larger typical mispricings  
• Higher curves = that ticker has more small edges  
""")

    # Close figures to prevent memory issues
    plt.close('all')

# ----- Summary Table by Ticker (Before Ticker Sections) -----
st.markdown("---")
st.markdown('<a name="summary-table-by-ticker-past-7-days"></a>', unsafe_allow_html=True)
st.markdown("**Summary Table by Ticker (Past 7 Days)**")
st.markdown("This summary table shows the number of logs for each ticker and the std, min, and max edges for easy review.")

recent_log_df = load_and_filter_log()
if not recent_log_df.empty:
    summary = recent_log_df.groupby('symbol').agg(
        count=('edge', 'size'),
        mean_edge=('edge', 'mean'),
        std_edge=('edge', 'std'),
        min_edge=('edge', 'min'),
        max_edge=('edge', 'max')
    ).reset_index()
    summary = summary.round(4)
    st.dataframe(summary, use_container_width=True)
else:
    st.info("No mispricings have been logged in the past 7 days. Run the scanner to collect data.")

# ----- Distribution Insights (Right after Summary Table) -----
with st.expander("📊 Click to view Chart and Distribution Insights", expanded=False):
    recent_log_df = load_and_filter_log()
    if not recent_log_df.empty:
        analyze_patterns(recent_log_df)
    else:
        st.info("Distribution insights will appear after running the scanner and collecting mispricing data over time.")

st.markdown("---")
st.markdown("### Next Section Overview:")
st.markdown("""
The scanner pulls live Yahoo Finance options data for each ticker and uses put-call parity to calculate synthetic stock prices: 

`Synthetic = Call Mid - Put Mid + Strike`. 

It then compares this to the actual stock price to find mispricings (edges).

Positive edges indicate the stock trades above its synthetic price, while negative edges suggest the opposite. 
            
The left tables show live data ranked by largest edges, and the right tables show historical mispricings logged when edges exceed $0.10.
""")

# ----- Auto-run Scan for All Tickers -----

log_path = "mispricing_log.csv"
tickers = list(ticker_names.keys())

for symbol in tickers:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    readable_name = ticker_names[symbol]
    # Add anchor tag for navigation
    readable_short = readable_name.split(" (")[0].lower().replace(' ', '-')
    st.markdown(f'<a name="{symbol.lower()}-{readable_short}"></a>', unsafe_allow_html=True)
    st.subheader(f"{symbol} — {readable_name}")

    # Start columns BEFORE rendering anything else
    cols = st.columns([1, 1])

    # Left Column: Fetch, Show Spot + Expiries + Live Table
    with cols[0]:
        df = fetch_chain(symbol)
        if df.empty:
            st.write(f"No data available for {symbol}.")
            continue
        
        # Filter for future expiries only
        df = df[pd.to_datetime(df['expiry']) >= pd.Timestamp.today()]
        if df.empty:
            st.write(f"No future expiries available for {symbol}.")
            continue
            
        df['ticker'] = symbol
        log_mispricings(df, symbol, threshold=0.10)

        # Display top 10 by edge
        display_df = df[['expiry', 'strike', 'synthetic_price', 'stock_price', 'edge', 'option_link']].head(10).copy()
        display_df['synthetic_price'] = display_df['synthetic_price'].round(4)
        display_df['stock_price'] = display_df['stock_price'].round(4)
        display_df['edge'] = display_df['edge'].round(4)
        display_df.rename(columns={
            'synthetic_price': 'Synthetic Px',
            'stock_price': 'Spot Px',
            'edge': 'Edge ($)',
            'option_link': 'Yahoo Finance'
        }, inplace=True)
        display_df = display_df[['expiry', 'strike', 'Synthetic Px', 'Spot Px', 'Edge ($)', 'Yahoo Finance']]
        st.markdown("**Live Option Chain (Top 10 by Edge)**")
        st.dataframe(display_df, height=350, use_container_width=True, 
                    column_config={
                        "Yahoo Finance": st.column_config.LinkColumn(
                            "Yahoo Finance",
                            help="View option chain on Yahoo Finance",
                            display_text="🔗 View"
                        )
                    })

    # Right Column: Logged Mispricings
    with cols[1]:
        # Add spacing to align with left column's spot price/expiries display
        st.markdown("&nbsp;", unsafe_allow_html=True)  # Match the spot price line height
        
        recent_log_df = load_and_filter_log()
        if not recent_log_df.empty:
            ticker_log_df = recent_log_df[recent_log_df['symbol'] == symbol].copy()
            ticker_log_df = ticker_log_df.sort_values(by='logged_at', ascending=False)
            ticker_log_df['edge'] = ticker_log_df['edge'].round(4)
            ticker_log_df.rename(columns={
                'logged_at': 'Logged At (UTC)',
                'expiry': 'Expiry',
                'strike': 'Strike',
                'mid_call': 'Mid Call',
                'mid_put': 'Mid Put',
                'synthetic_price': 'Synthetic Px',
                'stock_price': 'Spot Px',
                'edge': 'Edge ($)'
            }, inplace=True)
            show_cols = [
                'Logged At (UTC)', 'Expiry', 'Strike', 'Synthetic Px', 'Spot Px', 'Edge ($)'
            ]
            show_cols = [c for c in show_cols if c in ticker_log_df.columns]
            st.markdown("**Logged Mispricings (Past 7 Days)**")
            if not ticker_log_df.empty:
                st.dataframe(ticker_log_df[show_cols].head(10), height=350, use_container_width=True)
            else:
                st.info("No mispricings logged for this ticker in the past 7 days.")
        else:
            st.info("No mispricings have been logged yet. Try running the scanner.")


# ----- Footer -----
st.markdown("---")
st.caption("For educational and demonstration purposes only. Not investment advice.")