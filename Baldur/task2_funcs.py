import jellyfish
import numpy as np
import pandas as pd

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
    
    
# # For training word embeddings and clustering
# from gensim.models import Word2Vec
# from sklearn.cluster import KMeans

# # For faster computation
# import multiprocessing
# cores = multiprocessing.cpu_count()

# # A function for converting restaurants into embedding vectors based on its categories
# def resto2vec(tokens, w2v_model, normalize = True):
#     """Returns the embedding of a restaurant-related business as the mean of the tokens/words embeddings of its categories."""
    
#     embeddings = []
    
#     for token in tokens:
#         try:
#             embeddings.append(w2v_model.wv.get_vector(token, norm=normalize))
#         except KeyError:
#             continue
    
#     return np.array(embeddings).mean(axis=0) if len(embeddings) > 0 else np.zeros(shape = w2v_model.vector_size)

# def get_resto_embeddings(restaurant_cats_df, w2v_model):

#     # Creating word embedding model for restaurant categories
#     resto_names = restaurant_cats_df["name"]
#     resto_cats = restaurant_cats_df["categories"].to_list()

#     # Creating embedding for each restaurant
#     embeddings_df = {}
    
#     for resto, cat_list in zip(resto_names, resto_cats):
#         embeddings_df[resto] = resto2vec(cat_list, w2v_resto)
        
#         # Removing restaurant-related businesses whose all categories appear too few times in the corpus
#         if all(embeddings_df[resto] == np.zeros(shape = w2v_model.vector_size)):
#             del(embeddings_df[resto])
            
#     embeddings_df = pd.DataFrame({"restaurant": embeddings_df.keys(), 
#                                   "embeddings": embeddings_df.values()})
    
#     return w2v_resto, embeddings_df

# # The pipeline of the clustering task
# def cluster_restaurants(embeddings_df, k, seed_clus = 1):
        
#     # Fitting K-means
#     k_means = KMeans(n_clusters = k, random_state = seed_clus)
#     k_means.fit(np.vstack(embeddings_df["embeddings"]))
#     embeddings_df["cluster"] = k_means.labels_
    
#     return k_means