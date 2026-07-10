import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-mixture-of-depths-results"

WINDOWS = [63, 252, 504, 1008, 2016, 4032, 4536]

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# All available macro variables
MACRO_VARS = [
    "VIX", "DXY", "T10Y2Y", "TBILL_3M",
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS7",
    "DGS10", "DGS20", "DGS30"
]

# Mixture of Depths parameters
HIDDEN_SIZE = 64
NUM_LAYERS = 4
NUM_HEADS = 4
SEQ_LEN = 10
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 16
DEPTH_THRESHOLD = 0.5       # gating threshold for early exit
TOP_N = 3
