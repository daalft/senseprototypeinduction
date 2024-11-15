import os
import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import Counter
from WordTransformer import WordTransformer,InputExample

class PrototypeInduction:
    def __init__(self, iteration_count=51):
        self.enc_model = WordTransformer('pierluigic/xl-lexeme')
        self.cache = {}
        self.iteration_count = iteration_count
        
    def merge_it(self, arrs):
        if len(arrs) < 2:
            return arrs
        arr_agg = arrs[0]
        for arr in arrs[1:]:
            arr_agg = self.merge(arr_agg, arr)
        return arr_agg
    
    def my_sim_fn(self, a, b):
        u = np.asarray(a, dtype=float)
        v = np.asarray(b, dtype=float)
        if not np.any(u) or not np.any(v):
            return 0.0
        dot_product = np.dot(u, v)
        norm_u = np.sqrt(np.dot(u, u))
        norm_v = np.sqrt(np.dot(v, v))
        return dot_product / (norm_u * norm_v)

    def similarities(self, induced_sense, step_senses, similarity_function=None):
        if not similarity_function:
            similarity_function = self.my_sim_fn
        if len(step_senses) < 1:
            return [-1]
        return [similarity_function(induced_sense, step_sense) for step_sense in step_senses]

    def merge(self, step_sense, induced_sense):
        return np.mean(np.array([step_sense, induced_sense]), axis=0)

    def induction_step(self, ctx_idx_ids, direction="forward", step_threshold=0.7):
        step_senses = []
        # reverse ctx array if direction is backwards
        if direction == "backward":
            ctx_idx_ids = ctx_idx_ids[::-1]
        # shuffle ctx array if direction is random
        if direction == "random":
            random.shuffle(ctx_idx_ids)
        for ctx, idx, id_ in ctx_idx_ids:
            induced_sense = None
            if id_ in self.cache:
                induced_sense = self.cache[id_]
            else:
                input_ = InputExample(texts=ctx, positions=idx)
                induced_sense = self.enc_model.encode(input_)
                self.cache[id_] = induced_sense
            # similar to any bronze senses?
            step_similarities = self.similarities(induced_sense, step_senses)
            max_step_similarity = max(step_similarities)
            max_step_similarity_index = np.argmax(step_similarities)
            if max_step_similarity < step_threshold:
                step_senses.append(induced_sense)
            else:
                # merge induced and max
                induced_prototype = self.merge(step_senses[max_step_similarity_index], induced_sense)
                step_similarities[max_step_similarity_index] = induced_prototype
        return step_senses

    def merge_similar(self, arr1, arr2, similarity_function=None):
        if not similarity_function:
            similarity_function = self.my_sim_fn
        a = np.empty((len(arr1), len(arr2)))
        for i, x in enumerate(arr1):
            for j, y in enumerate(arr2):
                v = 0
                if i != j:
                    v = similarity_function(x, y)
                a[i][j] = v
        max_i, max_j = np.unravel_index(np.argmax(a, axis=None), a.shape)
        induced_sense = self.merge(arr1[max_i], arr2[max_j])
        return induced_sense

    def find_similar(self, arr1, arr2, similarity_function=None):
        if not similarity_function:
            similarity_function = self.my_sim_fn
        a = np.empty((len(arr1), len(arr2)))
        for i, x in enumerate(arr1):
            for j, y in enumerate(arr2):
                v = 0
                if i != j:
                    v = similarity_function(x, y)
                a[i][j] = v
        max_i, max_j = np.unravel_index(np.argmax(a, axis=None), a.shape)
        return arr1[max_i], arr2[max_j]

    def find_similar_indices(self, arr1, arr2, similarity_function=None):
        if not similarity_function:
            similarity_function = self.my_sim_fn
        a = np.empty((len(arr1), len(arr2)))
        for i, x in enumerate(arr1):
            for j, y in enumerate(arr2):
                v = 0
                if i != j:
                    v = similarity_function(x, y)
                a[i][j] = v
        max_i, max_j = np.unravel_index(np.argmax(a, axis=None), a.shape)
        return max_i, max_j

    def double_step(self, ctx_idx_ids):
        forward_senses = self.induction_step(ctx_idx_ids, direction="forward")
        backward_senses = self.induction_step(ctx_idx_ids, direction="backward")
        induced_sense = self.merge_similar(forward_senses, backward_senses)
        return induced_sense
    
    def matrix(self, arr1, arr2, similarity_function=None):
        if not similarity_function:
            similarity_function = self.my_sim_fn
        a = np.empty((len(arr1), len(arr2)))
        for i, x in enumerate(arr1):
            for j, y in enumerate(arr2):
                v = 0
                if i != j:
                    v = similarity_function(x, y)
                a[i][j] = v
        return a
    
    def calculate_mode(self, numbers):
        counts = Counter(numbers)
        max_count = max(counts.values())
        modes = [num for num, count in counts.items() if count == max_count]
        return modes
    
    def find_most_similar_indices(self, sense_arrays):
        num_arrays = len(sense_arrays)
        if num_arrays == 0:
            return []
        # Length of each sense_dist array
        array_length = len(sense_arrays[0])
        most_similar_indices = []
        # Iterate over each array
        for i in range(num_arrays):
            current_array = sense_arrays[i]
            # Initialize a list to store the maximum similarity for each element in the current array
            max_similarities = []
            # Iterate over each element in the current array
            for j in range(array_length):
                current_element = current_array[j]
                # Find the most similar element across all other arrays
                max_similarity = -1
                most_similar_index = (0,0)
                for k in range(num_arrays):
                    if k == i:
                        continue
                    other_array = sense_arrays[k]
                    for l in range(array_length):
                        other_element = other_array[l]
                        similarity = self.my_sim_fn(current_element, other_element)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_index = (k, l)
                # Append the index of the most similar element across arrays
                max_similarities.append(most_similar_index)
            # Find the index of the element with the highest similarity in the current array
            best_index = np.argmax([self.my_sim_fn(current_array[j], sense_arrays[most_similar_index[0]][most_similar_index[1]])
                                   for j, most_similar_index in enumerate(max_similarities)])
            most_similar_indices.append(best_index)
        return most_similar_indices
    
    def label_step(self, ctx_idx_ids, senses_with_id, return_value="id2label"): # return values: labels, ...
        labels = []
        senses = [x[0] for x in senses_with_id]
        lbx = [x[1] for x in senses_with_id]
        i2l = {}
        for _, _, id_ in ctx_idx_ids:
            # assume that the id is in the cache
            embedding = self.cache[id_]
            sims = self.similarities(embedding, senses)
            sim_idx = np.argmax(sims)
            label = lbx[sim_idx]
            labels.append(label)
            i2l[id_] = label
        if return_value == "id2label":
            return i2l
        return labels
    
    def full_induction(self, ctx_idx_ids, step_threshold):
        sense_arrays = []
        for _ in range(self.iteration_count):
            sense_arrays.append(self.induction_step(ctx_idx_ids, direction="random", step_threshold=step_threshold))
        # find most common number of senses
        mode = self.calculate_mode([len(x) for x in sense_arrays])[0]
        # retain only senses where the number of elements equals the mode
        sense_arrays_mode = [sub_array for sub_array in sense_arrays if len(sub_array) == mode]
        sense_id = 0
        final_senses_with_id = []
        # merge similar senses across arrays
        while len(sense_arrays_mode[0]) > 0:
            # find most similar indices across senses
            msis = self.find_most_similar_indices(sense_arrays_mode)
            # aggregate
            agg2 = []
            for i, msi in enumerate(msis):
                el = sense_arrays_mode[i].pop(msi)
                agg2.append(el)
            merged = self.merge_it(agg2)
            final_senses_with_id.append((merged, sense_id))
            sense_id += 1
        id2label = self.label_step(ctx_idx_ids, final_senses_with_id, return_value="id2label")
        ctx_idx_ids2 = []
        for ctx, idx, id_ in ctx_idx_ids:
            l = id2label[id_]
            obj = (ctx, idx, id_, l)
            ctx_idx_ids2.append(obj)
        return ctx_idx_ids2