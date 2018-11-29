import logging

import numpy as np

import settings
import torch
import torch.nn as nn


def select_model(args):
    if args.mil:
        return SelectionModuleMIL(args.each_input_size, args.num_templates, args.hidden_unit_list, args.embed_size, args.use_ids)
    else:
        return SelectionModule(args.each_input_size*args.num_templates, args.num_templates+1, args.hidden_unit_list, args.embed_size, args.use_ids)


class SelectionModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_unit_list, embed_size, use_ids):
        super(SelectionModule, self).__init__()
        module_list = []
        prev = embed_size + input_size if self.use_ids else input_size
        for i in range(len(hidden_unit_list)):
            module_list.append(nn.Sequential(
                nn.Linear(prev, hidden_unit_list[i]), nn.ReLU()))
            prev = hidden_unit_list[i]

        module_list.append(nn.Sequential(
            nn.Linear(prev, output_size), nn.Softmax(dim=1)))
        self.mlp = nn.Sequential(*module_list)
        if settings.cuda:
            self = self.cuda()
        logging.info('Created a normal concatenation model')

    def forward(self, x):
        x = x.float()
        return self.mlp(x)


class SelectionModuleMIL(nn.Module):
    def __init__(self, input_size, num_templates, hidden_unit_list, embed_size, use_ids):
        super(SelectionModuleMIL, self).__init__()
        module_list = []
        self.use_ids = use_ids
        prev = embed_size + input_size if self.use_ids else input_size
        for i in range(len(hidden_unit_list)):
            module_list.append(nn.Sequential(
                nn.Linear(prev, hidden_unit_list[i]), nn.ReLU()))
            prev = hidden_unit_list[i]

        module_list.append(nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid()))
        self.mlp = nn.Sequential(*module_list)
        self.num_templates = num_templates
        self.input_size = input_size
        self.embed_size = embed_size
        self.others_template_parameters = nn.Parameter(
            torch.randn(self.input_size))
        if settings.cuda:
            self = self.cuda()
        logging.info('Created an MIL model')

    def forward(self, x):
        x = x.float()
        if self.use_ids:
            embeds = x[:, :self.embed_size]
        x_vec = x[:, self.embed_size:] if self.use_ids else x
        ts = []
        for i in range(self.num_templates):
            if self.use_ids:
                ts.append(self.mlp(torch.cat(
                    (embeds, x_vec[:, i*self.input_size: (i+1)*self.input_size]), dim=1)))
            else:
                ts.append(
                    self.mlp(x_vec[:, i*self.input_size: (i+1)*self.input_size]))

        if self.use_ids:
            others_score = self.mlp(torch.cat(
                (embeds, self.others_template_parameters.repeat(embeds.size(0), 1)), dim=1))
        else:
            others_score = self.mlp(self.others_template_parameters)
        template_scores = torch.cat(ts, dim=1)
        template_scores = torch.cat((others_score.expand(
            template_scores.size(0), 1), template_scores), dim=1)

        return template_scores
