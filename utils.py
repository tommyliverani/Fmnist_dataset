import collections
import torch
import numpy as np
import tensorflow as tf

#transform a tensorflow tensor in to a pytorch tensor
def transform(t_tensor):
  return torch.from_numpy(np.array(t_tensor.numpy()))

#create a list of pytorch tensor
#collection: collection of dictionay{feature: tensor} 
#feature: feature used in the dictionart
def transform_collection(collection,feature):
  dataset=[]
  for row in collection:
    dataset.append(transform(row[feature]))
  return dataset


#create a dictionary of pytorch tensor
def create_dictionary(collection,features):
  d={}
  for feature in features:
    d[feature]=transform_collection(collection,feature)
  return d