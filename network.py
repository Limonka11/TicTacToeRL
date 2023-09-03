import torch.nn.functional as F

from torch import nn

class SimpleNN(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(SimpleNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		prediction = self.layer3(activation2)

		return prediction