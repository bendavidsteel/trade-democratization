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


class RecurGraphAgent(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_node_output_features, num_graph_output_features):
        super().__init__()

        conv_layer_size = 32
        lstm_layer_size = 32

        # graph convolutional layer to create graph representation
        conv_lin = torch.nn.Linear(num_edge_features, num_node_features * conv_layer_size)
        self.conv = geo.nn.NNConv(num_node_features, conv_layer_size, conv_lin)

        # lstm to learn sequential patterns
        self.lstm = torch.nn.LSTM(conv_layer_size, lstm_layer_size, dropout=0.5)

        # initial trainable hidden state for lstm
        self.lstm_h_s = torch.nn.Linear(1, lstm_layer_size)
        self.lstm_c_s = torch.nn.Linear(1, lstm_layer_size)

        # graph pooling layer
        self.pool = geo.nn.GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(lstm_layer_size, 2*lstm_layer_size), torch.nn.ReLU(), torch.nn.Linear(2*lstm_layer_size, 1)))

        # final graph output
        self.final_graph_linear = torch.nn.Linear(lstm_layer_size, num_graph_output_features)

        # final linear layer to allow full expressivity for regression after tanh activation in lstm
        self.final_node_linear = torch.nn.Linear(lstm_layer_size, num_node_output_features)

    def reset(self, initial):
        self.new_seq = True
        self.initial = initial

    def forward(self, input, step=True):
        if step:
            # create graph representation
            graph_step = torch.nn.functional.relu(self.conv(input.x, input.edge_index, input.edge_attr))

            # recurrent stage
            # initial state of lstm is representation of target prior to this sequence
            if self.new_seq:
                self.new_seq = False
                self.hs = self.lstm_h_s(self.initial).unsqueeze(0)
                self.cs = self.lstm_c_s(self.initial).unsqueeze(0)

            lstm_output, (self.hs, self.cs) = self.lstm(graph_step.unsqueeze(0), (self.hs, self.cs))

        else:
            initial, sequence = input.initial, input.sequence
            
            # create graph representation
            graph_collection = []
            for idx in range(len(sequence)):
                x, edge_index, edge_attr = sequence[idx].x, sequence[idx].edge_index, sequence[idx].edge_attr
                graph_step = torch.nn.functional.relu(self.conv(x, edge_index, edge_attr))
                graph_collection.append(graph_step)
            # provide graph representations as sequence to lstm
            graph_series = torch.stack(graph_collection)

            # recurrent stage
            # initial state of lstm is representation of target prior to this sequence
            lstm_output, _ = self.lstm(graph_series, (self.lstm_h_s(initial).unsqueeze(0), self.lstm_c_s(initial).unsqueeze(0)))

        # get last outputi
        lstm_final_output = lstm_output[-1, :, :]

        graph_pool = self.pool(lstm_final_output, input.batch)
        final_graph = self.final_graph_linear(graph_pool)
        graph_flattened = final_graph.view(-1)

        final_node = self.final_node_linear(lstm_final_output)
        node_flattened = final_node.view(-1)

        return torch.nn.functional.softmax(node_flattened), torch.nn.functional.softmax(graph_flattened)




class RegressionGraphNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_outputs):
        super(RegressionGraphNet, self).__init__()
        # we will use the edge conditioned convolution (NNConv) as this allows more than 1 edge feature
        # while also not requiring inputs to be scaled between 0 and 1

        hidden_layer_size = 5

        # NNConv requires a layer to transform the dimensionality of edge features to the required size
        lin1 = torch.nn.Linear(num_edge_features, num_node_features * hidden_layer_size)

        # arbitrarily using 16 as hidden layer size
        self.conv1 = geo.nn.NNConv(num_node_features, hidden_layer_size, lin1)
        self.lin1 = torch.nn.Linear(hidden_layer_size, num_outputs)

    def forward(self, data):
        # weights are by default floats
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.lin1(x)

        # final activation is linear as this is for regression
        return x



class EncoderNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()

        conv_layer_size = 32
        self.lstm_layer_size = 32

        # graph convolutional layer to create graph representation
        conv_lin = torch.nn.Linear(num_edge_features, num_node_features * conv_layer_size)
        self.conv = geo.nn.NNConv(num_node_features, conv_layer_size, conv_lin)

        # lstm to learn sequential patterns
        self.lstm = torch.nn.LSTM(conv_layer_size, self.lstm_layer_size, dropout=0.5)

    def forward(self, sequence):
        # do entire sequence all at once
        batch_size = sequence[0].x.shape[0]

        # create graph representation
        graph_collection = []
        for idx in range(len(sequence)):
            x, edge_index, edge_attr = sequence[idx].x, sequence[idx].edge_index, sequence[idx].edge_attr
            graph_step = torch.nn.functional.relu(self.conv(x, edge_index, edge_attr))
            graph_collection.append(graph_step)
        # provide graph representations as sequence to lstm
        graph_series = torch.stack(graph_collection)

        # recurrent stage
        # we don't care about the output for the encoder, just the hidden state
        # input hidden state defaults to zeros which is fine
        _, final_hidden = self.lstm(graph_series)

        # final activation is relu as this is for regression and the metrics of this dataset are all positive
        return final_hidden



class DecoderNet(torch.nn.Module):
    def __init__(self, num_output_features):
        super().__init__()

        lstm_layer_size = 32

        # lstm to learn sequential patterns
        # auto-regressive so same num input features as final output features
        self.lstm = torch.nn.LSTM(num_output_features, lstm_layer_size, dropout=0.5)

        # final linear layer to allow full expressivity for regression after tanh activation in lstm
        self.final_linear = torch.nn.Linear(lstm_layer_size, num_output_features)

    def forward(self, input, hidden):
        # need to do each recurrent iteration at a time to allow teacher forcing

        # recurrent stage
        # initial state of lstm is representation of target prior to this sequence
        output, hidden = self.lstm(input, hidden)

        # final activation is relu as this is for regression and the metrics of this dataset are all positive
        return self.final_linear(output), hidden