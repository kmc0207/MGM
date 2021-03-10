import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import time
import logging
from sklearn.cluster import KMeans
from sklearn import proprocessing
from data import gen_data,read_origin_relation
from model import ClassifierModel
import pandas as pd
from settings import args,TASK_DICT,init_logging





def train(training_data, valid_data, vocabulary, embedding_dim, hidden_dim,











if __name__ == '__main__':
    if args.task == 'fewrel':
        training_data,testing_data,valid_data,all_relations,vocabulary,embedding = gen_data()
        print("finish gen_data, start train")
        train(training_data,valid_data,vocabulary,args.embedding_dim,args.hidden_dim,args.device,args.batch_size,args.learning_rate,args.model_dir,all_relations,model=None,epoch=args.train_epoch)
    if args.task == 'dbpedia':
        training_data, testing_data, valid_data, all_relations, vocabulary, embedding = gen_data()
        train(training_data, valid_data, vocabulary, args.embedding_dim, args.hidden_dim, args.device, args.batch_size,
              args.learning_rate, args.model_dir, all_relations, model=None, epoch=args.train_epoch)
