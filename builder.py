import os
import types

import template1
import template2
import template3
import template4
import template5


def build_template(template_id, kb, base_model, use_hard_scoring=True,
                   load_dir=None, dump_dir=None):
    """
    Builds a template object of a given id. Loads from
    "load_dir/<template_id>.pkl" if load_dir is not None. If it is None, then
    dump_dir should be not None. Saves table to "dump_dir/<template_id>.pkl"
    """
    assert ((load_dir != None) or (dump_dir != None)
            ), "Atleast load_dir or dump_dir should be not None"

    load_file = None if load_dir is None else os.path.join(load_dir, str(template_id)+".pkl")
    save_file = None if dump_dir is None else os.path.join(dump_dir, str(template_id)+".pkl")
    
    obj = getattr(globals()['template'+str(template_id)], 'Template'+str(
        template_id))(
            kb, base_model, use_hard_scoring,
            load_file,save_file)
    print("Created Template of id %d\n" %(template_id))
    return obj


def build_templates(idlist, kb, base_model, use_hard_scoring=True,
                    load_dir=None, dump_dir=None):
    """
    Builds a template object of a given idlist. Loads from
    "load_dir/<template_id>.pkl" if load_dir is not None. If it is None, then
    dump_dir should be not None. Saves table to "dump_dir/<template_id>.pkl"
    """

    assert isinstance(idlist, list), "Need a list"

    obj_list = [build_template(
        el, kb, base_model, use_hard_scoring, load_dir, dump_dir) for el in idlist]

    return obj_list
