
import os
import code
import torch


from model import MLP
from dataset import CustomDataSet


if __name__ == '__main__' :

  net = MLP( dim_in=512, dim_out=512)
  dataset = CustomDataSet( len=32768, batch_size=128, dim_data=512)

  # check if GPU is available
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  net = net.to(device)

  t_in = torch.rand( (16, 256)).to(device)
  t_out = net( t_in)

  