import torch
cuda=False

def set_settings(args):
    global cuda
    cuda=args.cuda and torch.cuda.is_available()    