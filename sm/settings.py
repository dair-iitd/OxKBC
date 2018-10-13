import torch
cuda=False
templates=5

def set_settings(args):
    global cuda,templates
    cuda=args.cuda and torch.cuda.is_available()
    templates=args.templates