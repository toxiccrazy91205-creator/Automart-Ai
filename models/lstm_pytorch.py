import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# 🔹 1. Prepare time series data
def prepare_data(df):
    sales = df.groupby("Date")["Quantity_Sold"].sum().reset_index()

    values = sales["Quantity_Sold"].values.astype(float)

    return values


# 🔹 2. Create sequences (time steps)
def create_sequences(data, seq_length=7):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)


# 🔹 3. LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# 🔹 4. Train model
def train_model(df, epochs=10):
    data = prepare_data(df)

    X, y = create_sequences(data)

    # reshape for PyTorch
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    model = LSTMModel()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()

        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


# 🔹 5. Make prediction
def predict_future(model, df, seq_length=7):
    data = prepare_data(df)

    last_seq = data[-seq_length:]
    last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        prediction = model(last_seq)

    return prediction.item()


# 🔹 6. Full pipeline (main function)
def run_lstm(df):
    model = train_model(df)
    prediction = predict_future(model, df)

    return prediction


# 🔹 7. Test
if __name__ == "__main__":
    from utils.preprocessing import load_data, add_profit

    df = load_data()
    df = add_profit(df)

    pred = run_lstm(df)

    print(f"Next Day Sales Prediction: {pred}")