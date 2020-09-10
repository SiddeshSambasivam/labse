from sentence_transformers import SentenceTransformer, util
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
from tqdm import tqdm
from typing import List
import re

# PATH Variables
CONFIG_PATH = './checkpoints/bert_config.json'
CHECKPOINT_PATH = './checkpoints/bert_model.ckpt'
DICT_PATH = './checkpoints/vocab.txt'

class LaBSE:

    def __init__(self, CONFIG_PATH:str=CONFIG_PATH, CHECKPOINT_PATH:str=CHECKPOINT_PATH, DICT_PATH:str=DICT_PATH):
        '''Initiates the model and the tokenizer'''
        self.tokenizer = Tokenizer(DICT_PATH)
        self.model = build_transformer_model(
            CONFIG_PATH,
            CHECKPOINT_PATH,
            with_pool='linear'
        )

    def encode(self, sentences:List[str]):
        '''Embed the given list of sentences and return the embeddings '''
        embeddings = list()
        for sentence in tqdm(sentences):
            tokens_ids, segment_ids = self.tokenizer.encode(sentence)
            embed = self.model.predict([
                np.array([tokens_ids]), 
                np.array([segment_ids])
            ])[0]
    
            assert len(embed) == 768
            embeddings.append(embed)

        assert len(embeddings) == len(sentences)
        print(f'\n[INFO] Embedded {len(embeddings)} sentences\n')
        return embeddings
    

