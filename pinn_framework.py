# PINN architecture that combines a feedforward neural network 
# with a skip connection to improve gradient flow and model expressiveness. 
# The network uses Tanh activations and Xavier initialization, and is designed to learn solutions 
# to PDEs by minimizing residuals of governing equations and boundary/initial conditions.

class PINN(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super(PINN, self).__init__()
        layers = nn.ModuleList()

        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)

        self.skip = nn.Linear(in_features, out_features)
        self.skip.weight.data.normal_(std=0.01)
        self.skip.bias.data.zero_()

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.normal_(m.bias, std=0.01)

    def forward(self, x):
        identity = self.skip(x)
        outputs = self.net(x)
        return outputs + 0.1 * identity
