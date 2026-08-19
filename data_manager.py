import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(repo_id=config.DATA_REPO, filename="master_data.parquet", repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_returns_matrix(df, universe_tickers):
    returns = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    returns = returns.dropna(how='all')
    return returns

def get_universe_returns(universe_name, start_date=None, end_date=None):
    df = load_master_data()
    tickers = config.UNIVERSES.get(universe_name, [])
    returns = prepare_returns_matrix(df, tickers)
    if start_date:
        returns = returns[returns.index >= pd.to_datetime(start_date)]
    if end_date:
        returns = returns[returns.index <= pd.to_datetime(end_date)]
    return returns

def get_macro_data(start_date=None, end_date=None):
    df = load_master_data()
    macro_cols = [col for col in config.MACRO_VARS if col in df.columns]
    if not macro_cols:
        return None
    macro_df = df[macro_cols].copy()
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.sort_index()

    # IMPORTANT: FRED-style macro series (Treasury yields, VIX, DXY, etc.)
    # do not publish on exactly the same calendar as ETF prices -- different
    # holiday schedules, and some series update weekly rather than daily.
    # Left unfilled, this produces scattered NaNs in macro_df on dates where
    # the equity return index has valid data. A single NaN reaching the
    # model poisons the entire training run (NaN loss -> NaN gradients ->
    # NaN weights for the rest of that ticker's training), which is what was
    # producing identical 0.0 scores for every ticker/window/universe.
    # Forward-fill carries the last known macro reading forward (standard
    # practice for lower-frequency macro series), and a trailing bfill
    # handles any NaNs at the very start of the series.
    n_missing_before = int(macro_df.isna().sum().sum())
    macro_df = macro_df.ffill().bfill()
    n_missing_after = int(macro_df.isna().sum().sum())
    if n_missing_before > 0:
        print(f"[data_manager] Filled {n_missing_before} missing macro values "
              f"({n_missing_after} still missing after ffill/bfill).")

    if start_date:
        macro_df = macro_df[macro_df.index >= pd.to_datetime(start_date)]
    if end_date:
        macro_df = macro_df[macro_df.index <= pd.to_datetime(end_date)]
    return macro_df
