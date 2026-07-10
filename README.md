# Mixture of Depths (MoD) for ETFs

Implements Mixture of Depths (MoD) – a Transformer where tokens dynamically skip layers based on a learned depth gating mechanism. Tokens below the threshold exit early, achieving dynamic compute allocation per timestep. The score combines MoD prediction with momentum.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Depth gating with configurable threshold
- Dynamic compute allocation per token
- Score = MoD prediction × (1 + last_return)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-mixture-of-depths-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to neural net training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High positive score → ETF expected to rise tomorrow with dynamic compute allocation.
- Negative score → expected to fall.

## Requirements

See `requirements.txt`.
