
import code
import torch

class MLP( torch.nn.Module) :

  #####################
  def __init__( self, dim_in, dim_out, num_layers = 2, hidden_factor = 2,
                chs_in = 4, chs_out = 5,
                dropout_rate = 0., nonlin = torch.nn.GELU) :
 
    super( MLP, self).__init__()

    dim_hidden = int( dim_in * hidden_factor)

    self.layers = torch.nn.ModuleList()

    # Layer norm to improve stability of training
    self.layers.append( torch.nn.LayerNorm( dim_in))

    self.layers.append( torch.nn.Linear( dim_in, dim_hidden))
    self.layers.append( nonlin())
    self.layers.append( torch.nn.Dropout( p = dropout_rate))

    for il in range(num_layers-2) :
      self.layers.append( torch.nn.Linear( dim_hidden, dim_hidden))
      self.layers.append( nonlin())
      self.layers.append( torch.nn.Dropout( p = dropout_rate))
    
    self.layers.append( torch.nn.Linear( dim_hidden, dim_out))
    self.layers.append( nonlin())

    self.merge_channels = torch.nn.Linear(chs_in, chs_out)

  #####################
  def forward( self, x) :

    x_in = x

    for layer in self.layers :
      x = layer( x)

    x = self.merge_channels( x.transpose( -2, -1)).transpose( -2, -1)

    return x + x_in
