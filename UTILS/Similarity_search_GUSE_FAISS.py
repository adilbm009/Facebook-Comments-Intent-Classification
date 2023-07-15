import numpy as np
import faiss
import tensorflow_hub as tf
import tensorflow_hub as hub
guse_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        
#Faiss Training and Simiarity Search Function
def faiss_index_creation(probing_baselines):
    dim = 512
    index = faiss.IndexFlatIP(dim)
    index.add(guse_embed((probing_baselines)).numpy())
    return index

def search_faiss(df,probing_baselines,faiss_index,faiss_similarity_threshold):
    li_res_guss = []
    query= df.FeedText.tolist()
    dist, idx = faiss_index.search((guse_embed(query).numpy()),2)
    for i in range(len(query)):
        if np.max(dist[i]) >= faiss_similarity_threshold:
            li_res_guss.append((query[i], idx[i][np.argmax(dist[i])], np.max(dist[i])))
        else:
            li_res_guss.append((query[i], None, 0))
    return li_res_guss

