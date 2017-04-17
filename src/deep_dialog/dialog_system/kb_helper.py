
from fuzzywuzzy import fuzz
from deep_dialog import dialog_config
from collections import defaultdict


class KBHelper:
    """ An assistant to fill in values for the agent (which knows about slots of values) """

    def __init__(self, movie_dictionary, cmp_limit=90):
        """ Constructor for a KBHelper """

        self.cmp_limit = cmp_limit
        self.movie_dictionary = movie_dictionary
        self.cached_kb_slot = {}

    def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)

        Arguments:
        inform_slots_to_be_filled   --  Something that looks like {starttime:None, theater:None} where starttime and theater are slots that the agent needs filled
        current_slots               --  Contains a record of all filled slots in the conversation so far - for now, just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots

        Returns:
        filled_slots             --  A dictionary of form {slot1:value1, slot2:value2} for each sloti in inform_slots_to_be_filled
        """
        kb_results = self.available_results_from_kb(current_slots)
        current_slots = current_slots['inform_slots']  ## just already filled slots

        filled_slots = {}
        if 'taskcomplete' in inform_slots_to_be_filled.keys():
            filled_slots.update(current_slots)  ## TODO: check to be sure if it's ok

        #### TODO: very strange part of code begins here:

        for slot in inform_slots_to_be_filled.keys():
            if slot == 'numberofpeople':
                if slot in current_slots.keys():
                    filled_slots[slot] = current_slots[slot]
                else:
                    filled_slots[slot] = 'a lot of'  ## very bad to use this trick, but... :(

            if slot == 'ticket' or slot == 'taskcomplete':
                filled_slots[slot] = dialog_config.TICKET_AVAILABLE if len(
                    kb_results) > 0 else dialog_config.NO_VALUE_MATCH
                continue

            if slot == 'closing':
                continue

            ####################################################################
            #   Grab the value for the slot with the highest count and fill it
            ####################################################################

            values_counts = self.available_slot_values(slot, kb_results).items()
            filled_slots[slot] = max(values_counts, key=lambda x: x[1], default=(dialog_config.NO_VALUE_MATCH, 1))[
                0]

        return filled_slots

    @staticmethod
    def available_slot_values(slot, kb_results):
        """ Return the set of values available for the slot based on the current constraints """

        slot_values = defaultdict(int)
        for movie_id in kb_results:
            if slot in kb_results[movie_id]:
                slot_val = kb_results[movie_id][slot]
                slot_values[slot_val] += 1
        return slot_values

    def available_results_from_kb(self, current_slots):

        ret_result = []  ## what we should return;
        current_slots = current_slots['inform_slots']

        if len(current_slots) == 0:
            return self.movie_dictionary

        constrain_keys = current_slots.keys()

        ### leave only the keys which could be constraits ###
        constrain_keys = list(set(constrain_keys) & set(dialog_config.sys_inform_slots))
        constrain_keys = [k for k in constrain_keys if current_slots[k] != dialog_config.I_DO_NOT_CARE]

        if len(constrain_keys) == 0:
            return self.movie_dictionary

        ### starting looking for the suitable records:
        for id_ in self.movie_dictionary.keys():
            kb_keys = self.movie_dictionary[id_].keys()

            ### checking if all constraits is available
            ### for the current film id:

            if set(constrain_keys).issubset(set(kb_keys)):
                ### check if all constraints are the same in current_slots and in movie_dict:
                ### for that current film id:
                match = True
                for k in constrain_keys:
                    if fuzz.ratio(str(current_slots[k]), str(self.movie_dictionary[id_][k])) > self.cmp_limit:
                        continue
                    else:
                        match = False
                        break
                if match:
                    ret_result.append((id_, self.movie_dictionary[id_]))

        ### add everything to cache;
        query_idx_keys = frozenset(current_slots.items())
        self.cached_kb_slot.update({query_idx_keys: ret_result})

        ret_result = dict(ret_result)  ## It was dict in the previous version;
        return ret_result

    def database_results_for_agent(self, current_slots):
        """ A dictionary of the number of results matching each current constraint.
        The agent needs this to decide what to do next.
        Return the count statistics for each constraint in inform_slots """

        inform_slots = current_slots['inform_slots']
        kb_results = {key: 0 for key in inform_slots.keys()}
        kb_results['matching_all_constraints'] = 0

        query_idx_keys = frozenset(inform_slots.items())
        cached_kb_slot_ret = self.cached_kb_slot.get(query_idx_keys, [])

        if len(cached_kb_slot_ret) > 0:
            return cached_kb_slot_ret

        for movie_id in self.movie_dictionary.keys():
            all_slots_match = 1
            for slot in inform_slots.keys():
                if slot == 'ticket' or inform_slots[slot] == dialog_config.I_DO_NOT_CARE:
                    continue

                if slot in self.movie_dictionary[movie_id]:
                    if fuzz.ratio(inform_slots[slot], self.movie_dictionary[movie_id][slot]) > self.cmp_limit:
                        kb_results[slot] += 1
                    else:
                        all_slots_match = 0
                else:
                    all_slots_match = 0
            kb_results['matching_all_constraints'] += all_slots_match

        self.cached_kb_slot.update({query_idx_keys: kb_results})
        return kb_results

    def suggest_slot_values(self, request_slots, current_slots):
        """ Return the suggest slot values """

        avail_kb_results = self.available_results_from_kb(current_slots)
        return_suggest_slot_vals = {}
        for slot in request_slots.keys():
            avail_values_dict = self.available_slot_values(slot, avail_kb_results)
            values_counts = [(v, avail_values_dict[v]) for v in avail_values_dict.keys()]

            if len(values_counts) > 0:
                return_suggest_slot_vals[slot] = []
                sorted_dict = sorted(values_counts, key=lambda x: -x[1])
                for k in sorted_dict: return_suggest_slot_vals[slot].append(k[0])
            else:
                return_suggest_slot_vals[slot] = []

        return return_suggest_slot_vals