import os
import sys
import torch
sys.path.insert(1, '../') #BUG this is not a good solution
import friday.dataset.text

from friday.model import model, aux

def main():
    print("Friday debugging cli (fdc) starting...")
    #device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')
    device = torch.device('cpu')

    friday = model.Friday() #BUG: no parmas
main()