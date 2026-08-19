import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DepthGate(nn.Module):
    """
    Gating mechanism that decides whether a token continues to the next layer.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        gate_score = torch.sigmoid(self.gate(x))  # (batch, seq_len, 1)
        return gate_score

class MoDTransformerLayer(nn.Module):
    """
    Transformer layer with depth gating.
    Tokens can exit early if gate score is below threshold.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1, threshold=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = DepthGate(hidden_size)
        self.threshold = threshold

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Gate scores for each token
        gate_scores = self.gate(x)  # (batch, seq_len, 1)
        # Apply attention only to tokens that pass the gate (gating)
        # For simplicity, we apply attention to all tokens but weight by gate
        attn_out = self.self_attn(x, x, x)[0]
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        x = self.norm2(x + ffn_out)
        # Apply gate: tokens below threshold are zeroed out (early exit)
        # We keep the token but mask it so it doesn't affect later layers
        # But to implement early exit, we zero out tokens that exit
        mask = (gate_scores >= self.threshold).float()  # (batch, seq_len, 1)
        x = x * mask
        return x, mask

class MixtureOfDepths(nn.Module):
    """
    Transformer with Mixture-of-Depths: tokens dynamically skip layers.
    """
    def __init__(self, input_size, hidden_size=64, num_heads=4, num_layers=4, dropout=0.1, seq_len=10, threshold=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            MoDTransformerLayer(hidden_size, num_heads, dropout, threshold) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch, seq_len, _ = x.shape
        x = self.input_proj(x)
        # Track which tokens are active
        active_mask = torch.ones(batch, seq_len, 1, device=x.device)
        for layer in self.layers:
            x, layer_mask = layer(x)
            # Update active mask: a token is active if it passes all gates so far
            active_mask = active_mask * layer_mask
        # Pool over sequence: use only active tokens
        # If all tokens are inactive, fallback to mean
        if active_mask.sum() == 0:
            pooled = x.mean(dim=1)
        else:
            pooled = (x * active_mask).sum(dim=1) / (active_mask.sum(dim=1) + 1e-8)
        out = self.output_proj(pooled)
        return out.squeeze(-1)

def prepare_data(returns, macro_df, seq_len=10):
    """
    Prepare sequences for training.
    returns: pandas Series (single ETF) -- expected to already be standardized
    macro_df: pandas DataFrame (macro variables) -- expected to already be standardized
    """
    if len(returns) < seq_len + 1:
        return None, None
    common_idx = returns.index.intersection(macro_df.index)
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    # Drop any rows that are still NaN after alignment/standardization (e.g.
    # a ticker's pre-inception dates, or a macro column with no coverage at
    # all). Feeding a single NaN into the network poisons every weight for
    # the rest of training via NaN gradients -- silently, since nn.MSELoss()
    # and Adam don't raise on NaN, they just propagate it.
    valid_mask = ret_aligned.notna() & macro_aligned.notna().all(axis=1)
    ret_aligned = ret_aligned[valid_mask]
    macro_aligned = macro_aligned[valid_mask]
    if len(ret_aligned) < seq_len + 1:
        return None, None
    X, y = [], []
    for i in range(seq_len, len(ret_aligned)):
        ret_seq = ret_aligned.iloc[i-seq_len:i].values.reshape(-1, 1)
        macro_seq = macro_aligned.iloc[i-seq_len:i].values
        seq_features = np.concatenate([ret_seq, macro_seq], axis=1)
        X.append(seq_features)
        y.append(ret_aligned.iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def _standardize_series(s, eps=1e-8):
    """
    Z-score a pandas Series using its own window mean/std.
    Returns (scaled_series, mean, std).
    """
    mean = s.mean()
    std = s.std()
    if not np.isfinite(std) or std < eps:
        std = eps
    return (s - mean) / std, mean, std

def _standardize_df(df, eps=1e-8):
    """
    Z-score each column of a pandas DataFrame using its own window mean/std.
    Returns (scaled_df, mean_series, std_series).
    """
    mean = df.mean()
    std = df.std()
    std = std.where((std.notna()) & (std > eps), eps)
    scaled = (df - mean) / std
    return scaled, mean, std

def mod_score(returns, macro_df, hidden_size=64, num_heads=4, num_layers=4, dropout=0.1, seq_len=10, epochs=30, lr=0.001, batch_size=16, threshold=0.5):
    """
    Train Mixture-of-Depths model and return predicted next-day return with momentum enhancement.

    WHY THIS VERSION IS DIFFERENT
    ------------------------------
    1. Macro features (VIX, DXY, Treasury yields, etc.) live on very
       different numeric scales than a single ETF's daily return, and the
       macro block is IDENTICAL across every ticker in a given
       universe/window (14 of the 15 input columns). Feeding everything in
       raw/unscaled meant the model's gradients were dominated by the macro
       columns, so predictions could collapse toward the same near-zero
       value for every ticker. Fix: z-score every feature using that
       window's own statistics before it goes into the model. The momentum
       term still uses the true, unscaled last return.
    2. If any NaN reaches the model (macro data gaps, or a ticker's
       pre-inception dates), MSELoss/Adam silently propagate NaN through
       every weight for the rest of training -- every later prediction comes
       out NaN and gets masked to a "boring" 0.0 by the caller. This version
       drops NaN rows in prepare_data() and raises loudly if any NaN/Inf
       still makes it through, instead of returning a fake 0.0.
    """
    if len(returns) < seq_len + 1 or macro_df is None or macro_df.empty:
        return 0.0

    # Keep the raw last return for the momentum term (economically meaningful units).
    last_return_raw = returns.iloc[-1]

    # Standardize inputs on this window's own statistics so the single
    # ticker-specific return column isn't drowned out by the macro block.
    returns_scaled, _, _ = _standardize_series(returns)
    macro_scaled, _, _ = _standardize_df(macro_df)

    X, y = prepare_data(returns_scaled, macro_scaled, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        # Defensive guard: if NaN/Inf still made it through (e.g. an
        # upstream data issue we haven't seen yet), fail loudly here rather
        # than training a model to NaN weights and returning a value that
        # looks like a legitimate (if boring) score of 0.0 downstream.
        raise ValueError(
            "mod_score: NaN/Inf detected in model inputs after standardization "
            "and NaN-row filtering. Check the underlying master_data.parquet "
            "for gaps in this ticker's price history or the macro columns."
        )
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MixtureOfDepths(input_size, hidden_size, num_heads, num_layers, dropout, seq_len, threshold).to(device)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
    # Predict next day (using the same standardization as training)
    model.eval()
    with torch.no_grad():
        ret_seq = returns_scaled.iloc[-seq_len:].values.reshape(-1, 1)
        macro_seq = macro_scaled.iloc[-seq_len:].values
        last_seq = np.concatenate([ret_seq, macro_seq], axis=1)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(last_seq).item()
    # Momentum factor (uses the true, unscaled last return)
    momentum = 1.0 + last_return_raw
    momentum = max(0.5, min(2.0, momentum))
    final_score = pred * momentum
    return float(final_score)
