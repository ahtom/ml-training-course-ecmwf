
import torch

from model import MLP
from dataset import CustomDataSet


if __name__ == '__main__' :

  net = MLP( dim_in=512, dim_out=512)
  dataset = CustomDataSet( len=32768, batch_size=128, dim_data=512)

  # check if GPU is available
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  net = net.to(device)

  custom_dataset = CustomDataSet( len=32768, batch_size=128, dim_data=512)
  data_iter = iter(custom_dataset)

  lossfct = torch.nn.MSELoss()

  # load sample
  (source, target) = next(data_iter)
  source, target = source.to(device), target.to(device)
  # evaluate network
  pred = net( source)

  loss = lossfct( pred, target)

  # compute loss
  # idx = torch.arange( 512)
  # loss = lossfct( pred[idx], target[idx])
