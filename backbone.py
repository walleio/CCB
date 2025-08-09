import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from backbone_template import MLP, FC, ModelXtoC, End2EndModel

# independent and sequential model
def ModelXtoC_function(num_concepts, expand_dim):
    ModelXtoC_layer = ModelXtoC(num_concepts, expand_dim)
    return ModelXtoC_layer

# joint model
def ModelXtoCtoY_function(num_concepts, expand_dim):
    ModelXtoC_layer = ModelXtoC(num_concepts, expand_dim)
    ModelCtoY_layer = MLP(input_dim=num_concepts, expand_dim=expand_dim)
    return End2EndModel(ModelXtoC_layer, ModelCtoY_layer, num_concepts)