from hflayers import Hopfield
from torch.nn import Flatten, Linear, Module


class Friday(Module):
    def __init__(self, input_size, num_instances):
        super().__init__()
        self.hopfield = Hopfield(
            input_size=input_size,
            hidden_size=8,
            num_heads=8,
            update_steps_max=3,
            scaling=0.25)
        self.flatten = Flatten()
        self.output_projection = Linear(in_features=self.hopfield.output_size * num_instances, out_features=1)
        self.flatten2 = Flatten(start_dim=0)
    def forward(self, input):
        x = input
        x = self.hopfield(x)
        x = self.flatten(x)
        x = self.output_projection(x)
        x = self.flatten2(x)
        return x
