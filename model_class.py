import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc_soh = nn.Linear(hidden_size, output_size)
        self.fc_soc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        lstm_out = lstm_out.view(-1, self.hidden_size)  # Flatten the LSTM output
        output_soh = self.fc_soh(lstm_out)
        output_soc = self.fc_soc(lstm_out)
        return output_soh, output_soc
    