import torch.nn.functional as F

from torch import nn

class SimpleNN(nn.Module):
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim (int) - input dimensions
				out_dim (int) - output dimensions

			Return:
				None
		"""
		super(SimpleNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the NN.

			Parameters:
				obs (Tensor)- )bservations

			Return:
				prediction - The output of the model
		"""
		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		prediction = self.layer3(activation2)

		return prediction