import logging
import os
import types

import templates.template1
import templates.template2
import templates.template3
import templates.template4
import templates.template5
import templates.template6


def build_template(template_id, kblist, base_model, use_hard_scoring=True,
                   load_dir=None, dump_dir=None,parts = 1, offset = 0):
    """
    Builds a template object of a given id. Loads from
    "load_dir/<template_id>.pkl" if load_dir is not None. If it is None, then
    dump_dir should be not None. Saves table to "dump_dir/<template_id>.pkl"
    """
    assert ((load_dir != None) or (dump_dir != None)
            ), "Atleast load_dir or dump_dir should be not None"

    load_file = None if load_dir is None else os.path.join(
        load_dir, str(template_id)+".pkl")
    
    if parts == 1:
        save_file = None if dump_dir is None else os.path.join(
            dump_dir, str(template_id)+".pkl")
    else:
        save_file = None if dump_dir is None else os.path.join(
            dump_dir, '{}_p{}_o{}.pkl'.format(template_id, parts, offset))
    
    obj = getattr(getattr(globals()['templates'], 'template'+str(template_id)), 'Template'+str(template_id))(
        kblist, base_model, use_hard_scoring,
        load_file, save_file,parts,offset)
    logging.info("Created Template of id {}. Offset: {} of Total Parts: {}".format(template_id, offset, parts))
    return obj


def build_templates(idlist, kblist, base_model, use_hard_scoring=True,
                    load_dir=None, dump_dir=None, parts = 1, offset = 0):
    """
    Builds a template object of a given idlist. Loads from
    "load_dir/<template_id>.pkl" if load_dir is not None. If it is None, then
    dump_dir should be not None. Saves table to "dump_dir/<template_id>.pkl"
    """
    try:
        assert isinstance(idlist, list), "Need a list for tids"
        assert isinstance(kblist, list), "Need a list for kbs"
    except AssertionError as err:
        logging.exception("idlist should be a list")
        logging.exception("kblist should be a list")
        raise err

    obj_list = [build_template(
        el, kblist, base_model, use_hard_scoring, load_dir, dump_dir,parts, offset) for el in idlist]
    return obj_list
