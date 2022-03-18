from argparse import ArgumentError
from typing import List, Tuple
from hflayers import Hopfield
from torch import Tensor
from torch.nn import Flatten, Linear, Module
import logging


class Friday(Module):
    def __init__(self, input_size, num_instances):
        super().__init__()
        self.attentions:List[Tuple[Tensor, Tensor]] = list() # Tuple[query, state]
        self.hopfield = Hopfield(
            input_size=input_size,
            hidden_size=8,
            num_heads=8,
            update_steps_max=3,
            scaling=0.25)
        self.flatten = Flatten()
        self.output_projection = Linear(in_features=self.hopfield.output_size * num_instances, out_features=1)
        self.flatten2 = Flatten(start_dim=0)
    def ltminteraction(self, input:Tensor): # Data needs to be in the Friday ltm data format
        logging.info("INFO: Friday.data_load: calling data add function")
        x = input
        x = self.hopfield(x)
        x = self.flatten(x)
        x = self.output_projection(x)
        x = self.flatten2(x)
    def ltmload(self, input:Tensor):
        self.hopfield.train()
        self.ltminteraction(input)
    def ltmpull(self, query:Tensor):
        self.hopfield.eval()
    def ltmpredict(self, query:Tensor):
        self.hopfield.eval()
    
    def forward(self, query=None, data=None):
        logging.info("INFO: Friday.Model: model called")
        if data == None & query == None:
            logging.fatal("FATAL: Friday.forward: data and query both null")
            raise ArgumentError
        elif query == None:
            logging.info("INFO: Friday.forward: running query function")
            #TODO: load data
        elif data == None:
            logging.info("INFO: Friday.forward: calling data add function")
            #TODO: query data
        else:
            logging.info("INFO: Friday.forward: running data + query function")
            #TODO: query data using 
        x = query
        x = self.hopfield(x)
        x = self.flatten(x)
        x = self.output_projection(x)
        x = self.flatten2(x)
        return x
