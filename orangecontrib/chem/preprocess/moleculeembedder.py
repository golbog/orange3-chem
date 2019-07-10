import numpy as np

def filter_string_attributes(data):
    metas = data.domain.metas
    return [m for m in metas if m.is_string]


def pad_smiles(smiles, max_len):
    res = list()
    for smile in smiles:
        res.append(pad_smile(smile, max_len))
    return np.array(res)

def pad_smile(smile, max_len):
    if len(smile) < max_len:
        return smile + ' ' * (max_len - len(smile))
    return smile[:max_len]

def vectorize_smiles(smiles, charset, max_len=120):
    res = list()
    for smile in smiles:
        res.append(vectorize_smile(smile, charset, max_len))
    return np.array(res)

def vectorize_smile(smile, charset, max_len=120):
    x = np.zeros((max_len, len(charset)))
    for i, char in enumerate(pad_smile(smile, max_len)):
        # if char == ' ':
        #    continue
        try:
            x[i, charset[char]] = 1
        except KeyError:
            continue
    return x

def onehot_smiles(X, maxlen, charset):
    smiles = pad_smiles(X, maxlen)
    return vectorize_smiles(smiles, charset, maxlen)