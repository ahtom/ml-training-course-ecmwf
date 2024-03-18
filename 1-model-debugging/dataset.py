
import torch
import numpy as np

class CustomDataSet( torch.utils.data.IterableDataset):

  def __init__(self, len = 32768, batch_size = 128, dim_data = 512):
    
    super( CustomDataSet, self).__init__()

    self.len = len
    self.batch_size = batch_size
    self.dim_data = dim_data

  #####################
  def __iter__(self):

    iter_start, iter_end = self.worker_workset()

    for _ in range( iter_start, iter_end, self.batch_size) :

      batch = torch.rand( self.batch_size, self.dim_data)
      target = torch.rand( self.batch_size, self.dim_data)

      yield (batch, target)

  #####################
  def __len__(self):
      return self.len

  #####################
  def worker_workset( self) :

    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None: 
      iter_start = 0
      iter_end = len(self)

    else:  
      # split workload
      temp = len(self)
      per_worker = int( np.floor( temp / float(worker_info.num_workers) ) )
      worker_id = worker_info.id
      iter_start = int(worker_id * per_worker)
      iter_end = int(iter_start + per_worker)
      if worker_info.id+1 == worker_info.num_workers :
        iter_end = int(temp)

    return iter_start, iter_end