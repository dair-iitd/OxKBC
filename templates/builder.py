import logging
import os
import types

import templates.template1
import templates.template2
import templates.template3
import templates.template4
import templates.template5


def build_template(template_id, kb, base_model, use_hard_scoring=True,
                   load_dir=None, dump_dir=None):
    """
    Builds a template object of a given id. Loads from
    "load_dir/<template_id>.pkl" if load_dir is not None. If it is None, then
    dump_dir should be not None. Saves table to "dump_dir/<template_id>.pkl"
    """
    assert ((load_dir != None) or (dump_dir != None)
            ), "Atleast load_dir or dump_dir should be not None"

    load_file = None if load_dir is None else os.path.join(
        load_dir, str(template_id)+".pkl")
    save_file = None if dump_dir is None else os.path.join(
        dump_dir, str(template_id)+".pkl")

    obj = getattr(getattr(globals()['templates'], 'template'+str(template_id)), 'Template'+str(template_id))(
        kb, base_model, use_hard_scoring,
        load_file, save_file)
    logging.info("Created Template of id {0}".format(template_id))
    return obj


def build_templates(idlist, kb, base_model, use_hard_scoring=True,
                    load_dir=None, dump_dir=None):
    """
    Builds a template object of a given idlist. Loads from
    "load_dir/<template_id>.pkl" if load_dir is not None. If it is None, then
    dump_dir should be not None. Saves table to "dump_dir/<template_id>.pkl"
    """
    try:
        assert isinstance(idlist, list), "Need a list"
    except AssertionError as err:
        logging.exception("idlist should be a list")
        raise err

    obj_list = [build_template(
        el, kb, base_model, use_hard_scoring, load_dir, dump_dir) for el in idlist]

    return obj_list
