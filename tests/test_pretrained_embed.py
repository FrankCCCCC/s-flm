"""`utils.load_pretrained_word_embedding` across tokenizers (`model.pretrained_embed`).

A model whose tokenizer shares the data's vocabulary (gpt2-xl for gpt2) keeps its
embedding rows. A model with another vocabulary (BERT WordPiece) is mapped onto the
data's words: a GPT-2 word's row is the mean embedding of its WordPiece pieces, and
[UNK]'s when it has none. Needs the HF checkpoints (cached on the cluster).
"""
import pytest
import torch

from conftest import REPO_ROOT  # noqa: F401  (repo root on sys.path)
import utils

transformers = pytest.importorskip('transformers')
BERT = 'google-bert/bert-large-uncased'


def _source(name):
  model = transformers.AutoModel.from_pretrained(name)
  return (model.get_input_embeddings().weight.detach().float(),
          transformers.AutoTokenizer.from_pretrained(name))


def test_same_vocabulary_keeps_the_rows():
  emb, _ = _source('gpt2')
  out = utils.load_pretrained_word_embedding('gpt2', tokenizer_name='gpt2')
  assert torch.equal(out, emb)


def test_bert_rows_are_mapped_onto_gpt2_words():
  emb, bert = _source(BERT)
  gpt2 = transformers.AutoTokenizer.from_pretrained('gpt2')
  out = utils.load_pretrained_word_embedding(BERT, tokenizer_name='gpt2')
  assert out.shape == (len(gpt2), emb.shape[1])
  the = bert.convert_tokens_to_ids('the')
  for word in [' the', 'The']:   # uncased, no leading-space marker
    torch.testing.assert_close(out[gpt2.convert_tokens_to_ids(gpt2.tokenize(word))[0]], emb[the])
  pieces = bert.encode(' happ', add_special_tokens=False)   # ['ha', '##pp']
  v = gpt2.encode(' happ')
  assert len(v) == 1 and len(pieces) == 2
  torch.testing.assert_close(out[v[0]], emb[pieces].mean(0))
  torch.testing.assert_close(out[gpt2.encode('\n')[0]], emb[bert.unk_token_id])
  reduced = utils.load_pretrained_word_embedding(BERT, low_rank=48, tokenizer_name='gpt2')
  assert reduced.shape == (len(gpt2), 48) and torch.isfinite(reduced).all()
