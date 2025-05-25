import torch
import torch.nn as nn


class EncoderNetwork(nn.Module):
    """ Encoder network
        input_dim -> encoding_dim
    """

    def __init__(self, design_dim, observation_dim, hidden_dim=128, encoding_dim=64, hidden_depth=2, activation=nn.ReLU()):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.design_dim = design_dim
        input_dim = self.design_dim + observation_dim

        hidden_layers = []
        for _ in range(hidden_depth):
            hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            hidden_layers.append(activation)
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_dim, encoding_dim)


    def forward(self, xi, y):

        inputs = torch.cat([xi, y], dim=-1)

        x = self.hidden_layers(inputs)
        x = self.output_layer(x)
        return x


class EmitterNetwork(nn.Module):
    """ Emitter network 
        encoding_dim -> design_dim_flat
    """

    def __init__(self, encoding_dim, design_dim, hidden_dim=128, hidden_depth=2, activation=nn.Identity(), output_nonlinearity=nn.Identity()):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.design_dim = design_dim
        

        if hidden_depth > 0:
            hidden_layers = []
            for _ in range(hidden_depth):
                hidden_layers.append(nn.Linear(encoding_dim, hidden_dim))
                hidden_layers.append(activation)
                encoding_dim = hidden_dim
            self.hidden_layers = nn.Sequential(*hidden_layers)
            self.output_layer = nn.Linear(hidden_dim, design_dim)
        else:
            self.hidden_layers = nn.Identity()
            self.output_layer = nn.Linear(encoding_dim, design_dim)
        
        self.output_norm = output_nonlinearity

    def forward(self, r):
        x = self.hidden_layers(r)
        x = self.output_layer(x)
        x = self.output_norm(x)

        return x



class SetEquivariantDesignNetwork(nn.Module):
    """
    DAD (Encoder + Emitter)

    """
    def __init__(
        self,
        encoder_network,
        emission_network,
        dim_x,
        dim_y,
        empty_value
    ):
        super().__init__()
        self.encoder = encoder_network
        self.emitter = emission_network
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.register_buffer("prototype", empty_value.clone())
        self.register_parameter("empty_value", nn.Parameter(empty_value))

        # ensure the encoding_dim is the same in encoder and emitter
        assert self.encoder.encoding_dim == self.emitter.encoding_dim, "The encoding_dim in encoder and emitter must be the same!"


    def forward(self, xi, y):
        """ Generate design

        Args:
            xi [B, t, D]
            y [B, t, 1]

        """
        B, t, _ = xi.shape
        if t == 0:
            # Generate the first design
            # fill with empty value (expand to batch size)
            sum_encoding = self.empty_value.new_zeros(self.encoder.encoding_dim)
            # sum_encoding = self.empty_value.expand((B, -1))

        else:
            # Pooling
            sum_encoding = self.encoder(xi, y).sum(1)
        output = self.emitter(sum_encoding)

        return output
    
    @torch.no_grad()
    def run_trace(self, experiment, T, M):
        """ Run M parallel experiments and record the traces

        Args:
            experiment (BED): experiment simulator
            T (int): number of steps in a experiment trajectory
            M (int): number of rollouts
        """
        self.eval()

        theta = experiment.sample_theta((M, ))

        # history of an experiment
        xi_designs = torch.empty((M, T, self.dim_x))  # raw designs [M, T, D_x]
        y_outcomes = torch.empty((M, T, self.dim_y))  # [M, T, D_y]

        # T-steps experiment
        for t in range(T):
            xi = self.forward(xi_designs[:, :t], y_outcomes[:, :t])     # [B, D_x]
            y = experiment(experiment.to_design_space(xi), theta)                                   # [B, D_y]

            xi_designs[:, t] = xi
            y_outcomes[:, t] = y

        # convert designs to design space
        xi_designs = experiment.to_design_space(xi_designs)

        return theta, xi_designs, y_outcomes