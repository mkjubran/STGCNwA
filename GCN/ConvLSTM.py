import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size[0] // 2,
            bias=bias
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(1,1), bias=True):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, x):
        # x: [batch, time, channels, height, width]
        b, t, c, h, w = x.size()
        h_cur = torch.zeros(b, self.cell.hidden_dim, h, w, device=x.device)
        c_cur = torch.zeros(b, self.cell.hidden_dim, h, w, device=x.device)
        outputs = []

        for i in range(t):
            h_cur, c_cur = self.cell(x[:, i], h_cur, c_cur)
            outputs.append(h_cur.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [batch, time, hidden_dim, h, w]
