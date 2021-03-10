import os
import json
import argparse
import logging
import datetime
logger = logging.getLogger(__name__)
import GPUtil
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_VAL = -1
LEN_FACTOR = 1.163

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--data_dir",type=str, required=True)
    parser.add_argument("--train_epochs",type=int ,default=9)
    parser.add_argument("--seed" ,type=int,default=100)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_type",type=str,default='MultiTask')
    parser.add_argument("--model_dir",type=str,required=True)
    parser.add_argument("--embedding_dim",type=int,default=300)
    parser.add_argument("--hidden_dim",default=200)
    parser.add_argument("--batch_size",default=50)
    args = parser.parse_args()
    args.model_dir = os.path.join(args.model_dir,args.models_type,args.task)
    args.device_ids = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_ids)
    return args

def init_loggint(filename):
    logging_format ="%(asctime)s - %(update)s - %(relative)ss - %(levelname)s %(name)s -%(message)s"
    logging.basicConfig(format=logging_format,filename=filename,filemode='a',level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

args = parse_args()
TASK_DICT = {
    "fewrel" :{
        "train":os.path.join(args.data_dir,"fewrel_train_data"),
        "valid":os.path.join(args.data_dir,"fewrel_valid_data"),
        "test":os.path.join(args.data_dir,"fewrel_test_data"),
        "glove":os.path.join(args.data_dir,"fewrel_glove"),
        "relation" : os.path.join(args.data_dir,"glov_file")
    }
}