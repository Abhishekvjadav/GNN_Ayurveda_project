import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('app'))
from engine import GNNInferenceEngine
import torch
import numpy as np

eng = GNNInferenceEngine()
eng._load_fold(1)
embeddings = eng.fold_embeddings[1]
id_to_local = eng.node_id_to_local_maps[1]
model = eng.models[1]

c0 = 'COMPOUND_806ebd096539' # Agastyaprasha (11 ingredients)
c45 = 'COMPOUND_72dbc3a95fef' # Shatpala Ghrita (7 ingredients)

def get_h(cid):
    comp_local = id_to_local[cid]
    h_node = embeddings['Compound'][comp_local]
    ings = eng.compound_ingredients.get(cid, [])
    herb_vecs = []
    for hid in ings:
        if hid in id_to_local:
            h_local = id_to_local[hid]
            htype = eng.node_id_to_type.get(hid, 'Herb')
            if htype in embeddings and h_local < embeddings[htype].size(0):
                herb_vecs.append(embeddings[htype][h_local])
    if herb_vecs:
        h_ings = torch.stack(herb_vecs).mean(dim=0)
        return 0.5 * h_node + 0.5 * h_ings
    return h_node

h0 = get_h(c0)
h45 = get_h(c45)
r = model.decoder.r

print('Embedding diff:', torch.norm(h0 - h45).item())

s0 = [(h0 * r * embeddings['Disease'][id_to_local[d]]).sum().item() for d in eng.disease_ids]
s45 = [(h45 * r * embeddings['Disease'][id_to_local[d]]).sum().item() for d in eng.disease_ids]

r0 = np.argsort(s0)[::-1]
r45 = np.argsort(s45)[::-1]

d_names = [eng.node_id_to_canonical[d] for d in eng.disease_ids]
print('Top 5 Agastyaprasha:', [d_names[i] for i in r0[:5]])
print('Top 5 Shatpala Ghrita:', [d_names[i] for i in r45[:5]])
