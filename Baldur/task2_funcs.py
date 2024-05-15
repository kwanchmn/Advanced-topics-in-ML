import jellyfish
import numpy as np

# A function to replace a string based on Jaro scores
def replace_name(old_name, new_name, threshold, return_score = False):
    
    name = old_name
    jaro_score = jellyfish.jaro_similarity(old_name, new_name)
    
    # Replace old name with new one if similarity score is above or equal to the threshold
    if jaro_score >= threshold:
        name = new_name
    
    # Optional return of the Jaro score
    if return_score == True:
        return name, jaro_score
    
    return name

# A function to check
def jaro_replace_names(original_cats, new_name, threshold, return_ = False):
    
    original_cats = np.array(original_cats)
    replaced_cats = np.empty(len(original_cats), dtype = "object")
    jaro_scores = np.empty(len(original_cats))

    for i, cat in enumerate(original_cats):
        name, score = replace_name(cat, new_name, threshold, return_score = True)
        replaced_cats[i] = name
        jaro_scores[i] = score

    # Sorting the replacement from the highest Jaro score to the lowest
    score_sort_idx = np.argsort(jaro_scores)[::-1]

    # Printing the city names before and after replacement
    for old_cat, new_cat, score in zip(original_cats[score_sort_idx], replaced_cats[score_sort_idx], jaro_scores[score_sort_idx]):
        if score > threshold:
            print(f"{old_cat} -> {new_cat}, score: {score:.2f}")
    
    if return_ == True:
        return new_cat, score