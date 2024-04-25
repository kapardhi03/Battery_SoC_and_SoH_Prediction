import torch
from model_class import LSTMModel
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import numpy as np

model_info = torch.load('/Users/kapardhikannekanti/Battery_SoC_and_SoH_Prediction/lstm_model_info.pth') # replace with your local path
state_dict = torch.load('/Users/kapardhikannekanti/Battery_SoC_and_SoH_Prediction/lstm_model.pth')   # replace with your local path

model = LSTMModel(model_info['input_size'], model_info['hidden_size'], model_info['output_size'])
model.load_state_dict(state_dict)
model.eval()

criterion = nn.MSELoss()

def fit_scalers(X_train, y_train_soh, y_train_soc):
    scaler_X = MinMaxScaler()
    scaler_y_soh = MinMaxScaler()
    scaler_y_soc = MinMaxScaler()

    scaler_X.fit(X_train)
    scaler_y_soh.fit(y_train_soh.reshape(-1, 1))
    scaler_y_soc.fit(y_train_soc.reshape(-1, 1))

    return scaler_X, scaler_y_soh, scaler_y_soc

def predict_soh_soc(model, input_data, scaler_X, scaler_y_soh, scaler_y_soc):
    new_data_scaled = scaler_X.transform(input_data.values)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_soh, predicted_soc = model(new_data_tensor)

    predicted_soh_unscaled = scaler_y_soh.inverse_transform(predicted_soh.reshape(-1, 1))
    predicted_soc_unscaled = scaler_y_soc.inverse_transform(predicted_soc.reshape(-1, 1))

    return predicted_soh_unscaled.item(), predicted_soc_unscaled.item()