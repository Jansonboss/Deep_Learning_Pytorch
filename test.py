import pandas as pd
import numpy as np
import argparse

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

	def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
		super().__init__()
		# dir() will print all attributes
		# vars() will print all attribite with its valus (key value pairs)
		self.args = vars(args) if args is not None else {}

		input_dim = np.prod(data_config["input_dim"])
		num_classes = len(data_config["mapping"])

		default_FC1_DIM = 1024
		default_FC2_DIM = 128
		
		fc1_dim = self.args.get("fc1", default_FC1_DIM)
		fc2_dim = self.args.get("fc2", default_FC2_DIM)

		self.dropout = nn.Dropout(0.5)
		self.fc1 = nn.Linear(input_dim, fc1_dim)
		self.fc2 = nn.Linear(fc1_dim, fc2_dim)
		self.fc3 = nn.Linear(fc2_dim, num_classes)

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc3(x)
		return x
	

	def add_feature_readme_branch(cls):
		return "this the modified in the feature branch "
