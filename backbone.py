import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from backbone_template import MLP, ModelXtoC, End2EndModel

# joint model
def ModelXtoCtoY_function(num_concepts, expand_dim):
    ModelXtoC_layer = ModelXtoC(num_concepts=num_concepts, expand_dim=expand_dim)
    ModelCtoY_layer = MLP(input_dim=num_concepts, expand_dim=expand_dim)
    return End2EndModel(ModelXtoC_layer, ModelCtoY_layer, num_concepts)