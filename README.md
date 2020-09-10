# Language-agnostic BERT Sentence Embedding (LaBSE)

LaBSE model is a language agnostic sentence embedding model.

The model was trained on 109 languages to create a cross-lingual embedding space to embed any given sentence in any language to a common embedding space. It achieves the state-of-the-art performance on various bitext retrieval/mining tasks compareing with previous state-of-the-art with less language coverage.

## Installation

---

- ### Environment Setup

  For Unix Systems:

  1. Open Terminal and run the following bash command

     `pip install virtualenv`.

  2. Run the following command to create a virtualenv and install the project dependencies

     `python3 -m venv labse`

     `source labse/bin/activate`

     `pip install -r requirements.txt`

- ### Download the pretrained Weights

  The converted weights can be downloaded from the follwing links: [Link 1](https://pan.baidu.com/s/17qUdDSrPhhNTvPnEeI56sg) or [Link 2](https://drive.google.com/file/d/14Zaq8RE9NMyJb_9B-lkgFZQ9H1K-U-Nf)

  Finally, move the downloaded files to the checkpoints folder

## Sample Code

---

```python
>>> from model import LaBSE
>>> sentences = ['hello world', 'this is a sample case']
>>> model = LaBSE()
>>> embeddings = model.encode(sentences)

[INFO] Embedded 2 sentences
```

## Application

- Bitext Mining
- Semantic Similarity

---

## References

1. Paper: https://arxiv.org/abs/2007.01852
2. TFHUB: https://tfhub.dev/google/LaBSE/1

## Contact

- https://kexue.fm
