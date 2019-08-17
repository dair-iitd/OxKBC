import argparse
import datetime
import json
import os
import logging
import pickle
import utils
from IPython.core.debugger import Pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--template_save_dir',
                        required=False, default=None)
    parser.add_argument('--t_ids', nargs='+', type=int, required=True,
                        help='List of templates to build objects for')

    parser.add_argument('--log_level',
                        default='INFO',
                        dest='log_level',
                        type=utils._log_level_string_to_int,
                        nargs='?',
                        help='Set the logging output level. {0}'.format(utils._LOG_LEVEL_STRINGS))

    parser.add_argument('-p', '--parts', default=1, type=int,
                        help='How many parts do we need to collate?')

    args=parser.parse_args()
    logging.basicConfig(format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=args.log_level, datefmt='%d/%m/%Y %I:%M:%S %p')


    for tid in args.t_ids:
        global_dump=None
        flag=False
        missing_parts=[]
        for offset in range(args.parts):
            this_dump_file=os.path.join(
                args.template_save_dir, '{}_p{}_o{}.pkl'.format(tid, args.parts, offset))
            if not os.path.exists(this_dump_file):
                flag=True
                missing_parts.append(offset)
                break

            logging.info("Load from : {}".format(this_dump_file))
            with open(this_dump_file, "rb") as f:
                this_dump_dict=pickle.load(f)
            if global_dump is None:
                global_dump=this_dump_dict
            else:
                for key in ['table', 'stat_table']:
                    global_dump[key].update(this_dump_dict[key])
        #
        if flag:
            logging.error("Could not find all the parts for tid: {}. Missing parts: {}. NOT Creating dump".format(
                tid, ','.join(missing_parts)))
            continue

        global_dump_file=os.path.join(
            args.template_save_dir, '{}.pkl'.format(tid))
        logging.info("Creating dump for tid: {}. Dump File: {}".format(tid, global_dump_file))
        with open(global_dump_file,'wb') as gdf:
            pickle.dump(global_dump, gdf)

