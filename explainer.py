import utils
import numpy as np
import pandas
import collections
import logging
import os
import string


class Explainer:
    
    def __init__(self, dataset_root, kb, model, enum_to_id, rnum_to_id):

        self.WIKI_PREFIX_URL = 'https://en.wikipedia.org/wiki/'
        self.WIKI_NAME_TEMPLATE = '<a target=\"_blank\" href=\"' + self.WIKI_PREFIX_URL+'$wiki_id\">$name</a>'
        with open('css_style.css', 'r') as css:
            self.CSS_STYLE = css.read()
        self.NO_EXPLANATION = "No explanation for this fact"

        self.model = model
        self.enum_to_id = enum_to_id
        self.rnum_to_id = rnum_to_id

        self.e1_e2_r = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.e2_e1_r = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.r_e2_e1 = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for e1, r, e2 in kb.facts:
            self.e1_e2_r[e1][e2].append(r)
            self.e2_e1_r[e2][e1].append(r)
            self.r_e2_e1[r][e2].append(e1)
        logging.info("Initialized data structures for fast lookup")

        self.entity_names = self.read_entity_names(
            os.path.join(dataset_root, "mid2wikipedia_cleaned.tsv"))
        logging.info("Read entity names")
        self.relation_names = self.read_relation_names(
            os.path.join(dataset_root, "relation_names.txt"))
        logging.info("Read relation names and made heuristic purge for them")

    def read_relation_names(self, path):
        relation_names = {}
        with open(path, "r", errors='ignore', encoding='ascii') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split()
                if content[0] in relation_names:
                    logging.warn('Duplicate Entity found %s in line %s' %
                                 (content[0], ' '.join(line)))
                relation_names[content[0]] = ' '.join(content[1:])

        # heuristic_purge_relations:
        for _, r in self.rnum_to_id.items():
            if r in relation_names:
                continue
            parts = r.split(".")
            name = ""
            if (len(parts) == 1):
                parts = parts[0].split("/")
                name = parts[0]+" "+parts[-1]
            else:
                name = parts[0].split("/")[-2]+" "+parts[1].split("/")[-1]
            relation_names[r] = name

        return relation_names

    def read_entity_names(self, path):
        entity_names = {}
        with open(path, "r", errors='ignore', encoding='ascii') as f:
            lines = f.readlines()
            for line in lines:
                content_raw = line.split('\t')
                content = [el.strip() for el in content_raw]
                if content[0] in entity_names:
                    logging.warn('Duplicate Entity found %s in line %s' %
                                 (content[0], ' '.join(line)))
                name = content[1]
                wiki_id = content[2]
                entity_names[content[0]] = {"name": name, "wiki_id": wiki_id}
        return entity_names

    def similar_relations(self, list1, list2, hard_match, thresh=0.5):

        def hard(lis1, lis2):
            s1 = set(lis1)
            s2 = set(lis2)
            intersection = list(s1 & s2)
            pair_intersection = [(x, x) for x in intersection]
            return pair_intersection

        def soft(lis1, lis2, thresh):
            list_r1r2 = []
            if (len(lis1) > len(lis2)):
                lis1, lis2 = lis2, lis1
            for r1 in lis1:
                best_score = -1
                best_rel = -1
                for r2 in lis2:
                    score = self.model.get_relation_similarity(r1, r2)
                    if(score > best_score):
                        best_score = score
                        best_rel = r2
                if(best_rel != -1 and best_score > thresh):
                    list_r1r2.append((r1, best_rel))
            return list_r1r2

        if(hard_match):
            return hard(list1, list2)
        else:
            return soft(list1, list2, thresh)

    def explain_entity_similarity(self, e1, e2, hard_match):
        def explain_entity_similarity_aux(e1, e2, flip, look_up, hard_match=True):
            d1 = look_up[e1]
            d2 = look_up[e2]
            relevant_tuple = []

            for e in (set(d1.keys()) & set(d2.keys())):
                relevant = self.similar_relations(d1[e], d2[e], hard_match)
                for r1r2 in relevant:
                    if flip:
                        relevant_tuple.append(
                            ([e, r1r2[0], e1], [e, r1r2[1], e2]))
                    else:
                        relevant_tuple.append(
                            ([e1, r1r2[0], e], [e2, r1r2[1], e]))
            return relevant_tuple

        list1 = explain_entity_similarity_aux(
            e1, e2, False, self.e1_e2_r, hard_match)
        list2 = explain_entity_similarity_aux(
            e1, e2, True, self.e2_e1_r, hard_match)
        return list1, list2

    def freq_for_relation(self, r, e2):
        if r in self.r_e2_e1 and e2 in self.r_e2_e1[r]:
            return self.r_e2_e1[r][e2]
        return []

    def freq_for_entity(self, e1, e2):
        if e1 in self.e1_e2_r and e2 in self.e1_e2_r[e1]:
            return self.e1_e2_r[e1][e2]
        return []

    def get_cs_string(self, l):
        if len(l) <= 2:
            return ' , '.join(l)
        else:
            n = len(l) - 2
            string_more = '{} and <div class=\"tooltip2\">{} more...<span class=\"tooltiptext2\">'.format(
                ' , '.join(l[:2]), n)
            for el in l[2:]:
                string_more += el + "<br>"
            string_more += "</span></div>"
        return string_more

    def get_r_name(self, r):
        try:
            rnum = int(r)
            rid = self.rnum_to_id[rnum]
        except ValueError:
            rid = r
        return self.relation_names.get(rid, rid)

    def get_e_name(self, e, add_wiki=True):
        try:
            enum = int(e)
            eid = self.enum_to_id[enum]
        except ValueError:
            eid = e
        data = self.entity_names.get(eid, {"name": eid, "wiki_id": ""})
        if(add_wiki and data["wiki_id"] != ""):
            return string.Template(self.WIKI_NAME_TEMPLATE).substitute(**data)
        else:
            return data["name"]

    def name_fact(self, fact, add_wiki=True):
        e1_name = self.get_e_name(fact[0], add_wiki)
        r_name = self.get_r_name(fact[1])
        e2_name = self.get_e_name(fact[2], add_wiki)
        return [e1_name,r_name,e2_name]

    def html_fact(self,fact):
        fact_html_template = '<b>(<font color="blue">$e1</font>, <font color="green">$r</font>, <font color="blue">$e2</font>)</b>'
        named_fact = self.name_fact(fact)
        return string.Template(fact_html_template).substitute(e1=named_fact[0], r=named_fact[1], e2=named_fact[2])

    def get_relation_frequent(self, fact):
        r_name = self.get_r_name(fact[1])
        e2_name = self.get_e_name(fact[2])

        other_part = self.freq_for_relation(fact[1], fact[2])
        other_knowledge = [self.get_e_name(el) for el in other_part]

        string_frequent = "<div class=\"tooltip\">frequently seen <span class=\"tooltiptext\">"
        cs_string = self.get_cs_string(other_knowledge)
        string_frequent += "(" + cs_string + " ) " + r_name + "  " + e2_name + "<br>"
        string_frequent += "</span></div>"

        return string_frequent

    def get_entity_frequent(self, fact):
        e1_name = self.get_e_name(fact[1])
        e2_name = self.get_e_name(fact[2])

        other_part = self.freq_for_entity(fact[0], fact[2])
        other_knowledge = [self.get_r_name(el) for el in other_part]

        string_frequent = "<div class=\"tooltip\">frequently seen <span class=\"tooltiptext\">"
        cs_string = self.get_cs_string(other_knowledge)
        string_frequent += e1_name + \
            " ( " + cs_string + " ) " + e2_name + "<br>"
        string_frequent += "</span></div>"

        return string_frequent

    def get_entity_similar(self, e1, e2):
        tuples_similar_head, tuples_similar_tail = self.explain_entity_similarity(
            e1, e2, hard_match=True)
        e1_name = self.get_e_name(e1)
        e2_name = self.get_e_name(e2)

        rel_dir_head = collections.defaultdict(list)
        for t in tuples_similar_head:
            rel_dir_head[self.get_r_name(t[0][1])].append(self.get_e_name(t[0][2]))

        rel_dir_tail = collections.defaultdict(list)
        for t in tuples_similar_tail:
            rel_dir_tail[self.get_r_name(t[0][1])].append(self.get_e_name(t[0][0]))

        string_similar = "<div class=\"tooltip\">similar <span class=\"tooltiptext\">"
        for rel in rel_dir_head:
            cs_string = self.get_cs_string(rel_dir_head[rel])
            string_similar += e1_name + " and " + e2_name + " " + rel + " (" + cs_string + " )<br>"
        string_similar += "<br>"
        for rel in rel_dir_tail:
            cs_string = self.get_cs_string(rel_dir_tail[rel])
            string_similar += "(" + cs_string + " ) " + rel + "  " + e1_name + " and " + e2_name + "<br>"
        string_similar += "<br>"
        string_similar += "</span></div>"
        return string_similar