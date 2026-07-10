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
    returns: pandas Series (single ETF)
    macro_df: pandas DataFrame (macro variables)
    """
    if len(returns) < seq_len + 1:
        return None, None
    common_idx = returns.index.intersection(macro_df.index)
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
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

def mod_score(returns, macro_df, hidden_size=64, num_heads=4, num_layers=4, dropout=0.1, seq_len=10, epochs=30, lr=0.001, batch_size=16, threshold=0.5):
    """
    Train Mixture-of-Depths model and return predicted next-day return with momentum enhancement.
    """
    X, y = prepare_data(returns, macro_df, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MixtureOfDepths(input_size, hidden_size, num_heads, num_layers, dropout, seq_len, threshold).to(device)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    # Predict next day
    model.eval()
    with torch.no_grad():
        ret_seq = returns.iloc[-seq_len:].values.reshape(-1, 1)
        macro_seq = macro_df.iloc[-seq_len:].values
        last_seq = np.concatenate([ret_seq, macro_seq], axis=1)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(last_seq).item()
    # Momentum factor
    last_return = returns.iloc[-1]
    momentum = 1.0 + last_return
    momentum = max(0.5, min(2.0, momentum))
    final_score = pred * momentum
    return float(final_score)
