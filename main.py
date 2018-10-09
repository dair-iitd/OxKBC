import kb
import builder
import models
import argparse
import os
import datetime
import json
import utils


def main(dataset_root, model_weight_file, template_load_dir, template_save_dir,
         model_type, templates_idlist, introduce_oov, use_hard_scoring=True):
    ktrain = kb.KnowledgeBase(os.path.join(dataset_root, 'train.txt'))
    if introduce_oov:
        ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
    ktest = kb.KnowledgeBase(os.path.join(dataset_root, 'test.txt'), ktrain.entity_map, ktrain.relation_map,
                             add_unknowns=not introduce_oov)
    kvalid = kb.KnowledgeBase(os.path.join(dataset_root, 'valid.txt'), ktrain.entity_map, ktrain.relation_map,
                              add_unknowns=not introduce_oov)

    if(model_type == "distmult"):
        base_model = models.TypedDM(model_weight_file)
    elif(model_type == "complex"):
        base_model = models.TypedComplex(model_weight_file)
    else:
        raise "Invalid Model type given"

    templates_obj = builder.build_templates(templates_idlist, ktrain, base_model,
                                            use_hard_scoring, template_load_dir, template_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument(
        '-m', '--model_type', help="model name. Can be distmult or complex ", required=True)
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model wieghts", required=True)
    parser.add_argument('-l', '--template_load_dir',
                        required=False, default=None)
    parser.add_argument('-s', '--template_save_dir',
                        required=False, default=None)
    parser.add_argument('-v', '--oov_entity', required=False, default=True)
    parser.add_argument('--t_ids', nargs='+', type=int, required=True,
                        help='List of templates to train for')
    parser.add_argument('--data_repository_root',
                        required=False, default='data')
    args = parser.parse_args()

    dataset_root = os.path.join(args.data_repository_root, args.dataset)
    main(dataset_root, args.model_weights, args.template_load_dir,
         args.template_save_dir, args.model_type, args.t_ids, args.oov_entity)
