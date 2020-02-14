#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2020 Jerry Peng
#  mimics kaldi feat-to-post
# it convert feature-formatted posterior into Posterior-formatted posterior

#  feature-formatted posterior:
#    <uttid> [[0.1, 0.89, 0. , 0. , 0. , 0.01],
#             [0. , 0.9 , 0. , 0. , 0. , 0.10],
#               ...
#             [0.8, 0.2 , 0.0, 0.0, 0.0, 0.00]]
#      ...

#  Posterior-formatted posterior:
#    <uttid> [[(0,0.1), (1,0.89), (5,0.01)],
#             [(1,0,9), (5,0.1)],
#               ...
#             [(0,0.8), (1,0.2)]]
#      ...

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
# from deco import concurrent, synchronized
from kaldi.hmm import Posterior
from kaldi.util.table import SequentialMatrixReader, PosteriorWriter
import numpy as np


def feat_to_post(feature_rspecifier, posterior_wspecifier, top_n=10, rescale=False):
  assert top_n >= 1
  with SequentialMatrixReader(feature_rspecifier) as feature_reader, \
          PosteriorWriter(posterior_wspecifier) as posterior_writer:
    for uttid, feat in feature_reader:
      feat_np = feat.numpy()
      posts_lst = []
      assert top_n <= feat_np.shape[1]
      for row in feat_np:
        idxs = np.argpartition(row, -top_n)[-top_n:]
        if not rescale:
          post = [(int(idx), float(row[idx])) for idx in idxs]
        else:
          post_candidates = [float(row[idx]) for idx in idxs]
          sum_post = sum(post_candidates)
          if 0 == sum_post:
            post = [(int(idx), 1./len(idxs)) for idx in idxs]
          else:
            post = [(int(idx), post_candidates[idx]/sum_post) for idx in idxs]
        posts_lst.append(post)
      posterior_writer[uttid] = Posterior().from_posteriors(posts_lst)
  return True


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Convert feature-formatted posterior into Posterior-formatted posterior

   feature-formatted posterior:
     <uttid> [[0.1, 0.89, 0. , 0. , 0. , 0.01],
              [0. , 0.9 , 0. , 0. , 0. , 0.10],
                ... 
              [0.8, 0.2 , 0.0, 0.0, 0.0, 0.00]]
       ...

   Posterior-formatted posterior:
     <uttid> [[(0,0.1), (1,0.89), (5,0.01)],
              [(1,0,9), (5,0.1)],
                ...
              [(0,0.8), (1,0.2)]]
       ... 

  Usage: feat-to-post.py [options] feature_rspecifier posteriors_wspecifier

  e.g.
      feat-to-post scp:feats.scp ark:post.ark

  """
  po = ParseOptions(usage)
  po.register_int("top-n", 10,
                  "only keep highest N posteriors per frame, 10 by default")
  po.register_bool("rescale", False,
                   "rescale top N posteriors to let summation equals to 1, false by default")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  feature_rspecifier = po.get_arg(1)
  posterior_wspecifier = po.get_arg(2)
  isSuccess = feat_to_post(feature_rspecifier, posterior_wspecifier,
                           opts.top_n, opts.rescale)
  if not isSuccess:
    sys.exit()
