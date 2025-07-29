import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_ch = input_channels if i == 0 else hidden_channels
            self.layers.append(ConvLSTMCell(input_ch, hidden_channels, kernel_size))

    def forward(self, x):
        b, t, c, h, w = x.size()
        h_t = [torch.zeros(b, layer.hidden_channels, h, w, device=x.device) for layer in self.layers]
        c_t = [torch.zeros(b, layer.hidden_channels, h, w, device=x.device) for layer in self.layers]

        outputs = []
        for time_step in range(t):
            input = x[:, time_step, :, :, :]
            for i, layer in enumerate(self.layers):
                h_t[i], c_t[i] = layer(input, h_t[i], c_t[i])
                input = h_t[i]
            outputs.append(h_t[-1])

        outputs = torch.stack(outputs, dim=1)
        return outputs

