from ast import Raise
from torch import Tensor, nn

class FridayInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(input: Tensor, type:Tensor):
        if type == None:
            raise AttributeError #TODO: Add type detection 
        raise NotImplemented
        # Get input type info
        # Feed out junk data
        # find the topic then direct to relevant attention or create new one
