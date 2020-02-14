#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2020 Jerry Peng
#  count feature-formatted posteriors

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
# from deco import concurrent, synchronized
from kaldi.hmm import Posterior
from kaldi.util.table import SequentialMatrixReader, PosteriorWriter, VectorWriter
from kaldi.matrix import Vector
import numpy as np
# import os


def post_to_count(feature_rspecifier, cnt_wspecifier, normalize=False, per_utt=False):
  with SequentialMatrixReader(feature_rspecifier) as feature_reader, \
          VectorWriter(cnt_wspecifier) as cnt_writer:
      if per_utt:
        for uttid, feat in feature_reader:
          cnt_writer[uttid] = Vector(feat.numpy().mean(axis=0))
      else:
        vec = 0
        num_done = 0
        for uttid, feat in feature_reader:
          vec = vec + feat.numpy().mean(axis=0)
          num_done = num_done + 1
        if normalize:
          vec = vec / num_done
        cnt_writer[str(num_done)] = Vector(vec)
  return True


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Compute the counts of *feature-formatted* posterior for each mixture. 
  If --normalize=True and --per-utt=False, the counts will be averaged by the
    number of utterances.
  Usage: post-count.py [options] feature_rspecifier posteriors_wspecifier

  e.g.
      post-count scp:feats.scp ark,t:count.txt

  """
  po = ParseOptions(usage)
  po.register_bool("normalize", False, "normalize the counts, False by default")
  po.register_bool("per-utt", False, "Count per utterance, False by default")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  feature_rspecifier = po.get_arg(1)
  posterior_wspecifier = po.get_arg(2)
  isSuccess = post_to_count(feature_rspecifier, posterior_wspecifier, normalize=opts.normalize, per_utt=opts.per_utt)
  if not isSuccess:
    sys.exit()
