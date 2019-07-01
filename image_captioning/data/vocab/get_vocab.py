import pickle


def get_vocab(vocab_path):
    """
    get the vocab of the dataset
    :param vocab_path: the path of the vocab
    """
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
