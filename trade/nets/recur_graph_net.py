import torch
import torch_geometric as geo

class RecurGraphNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_output_features):
        super().__init__()

        conv_layer_size = 32
        lstm_layer_size = 32

        # graph convolutional layer to create graph representation
        conv_lin = torch.nn.Linear(num_edge_features, num_node_features * conv_layer_size)
        self.conv = geo.nn.NNConv(num_node_features, conv_layer_size, conv_lin)

        # lstm to learn sequential patterns
        self.lstm = torch.nn.LSTM(conv_layer_size, lstm_layer_size, dropout=0.5)

        # initial trainable hidden state for lstm
        self.lstm_h_s = torch.nn.Linear(num_output_features, lstm_layer_size)
        self.lstm_c_s = torch.nn.Linear(num_output_features, lstm_layer_size)

        # final linear layer to allow full expressivity for regression after tanh activation in lstm
        self.final_linear = torch.nn.Linear(lstm_layer_size, num_output_features)

    def reset(self, initial):
        self.initial = initial
        self.new_seq = True

    def forward(self, input):
        # create graph representation
        graph_step = torch.nn.functional.relu(self.conv(input.x, input.edge_index, input.edge_attr))

        # recurrent stage
        # initial state of lstm is representation of target prior to this sequence
        if self.new_seq:
            self.new_seq = False
            self.hs = self.lstm_h_s(self.initial).unsqueeze(0)
            self.cs = self.lstm_c_s(self.initial).unsqueeze(0)

        lstm_output, (self.hs, self.cs) = self.lstm(graph_step.unsqueeze(0), (self.hs, self.cs))

        return self.final_linear(lstm_output.squeeze(0))