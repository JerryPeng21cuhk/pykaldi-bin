#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2020 Jerry Peng

# This script use pca to convert high-dimension vectors into low-dimension vectors.

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialVectorReader, VectorWriter, read_script_file
import numpy as np
from sklearn.decomposition import PCA


def pca_vector(vector_rspecifier, vector_wspecifier, output_dim=2):
  vectors = []
  uttids = []
  with SequentialVectorReader(vector_rspecifier) as vector_reader:
    for uttid, vector in vector_reader:
      uttids.append(uttid)
      vectors.append(vector.numpy())
  # vectors is a set of row vectors indexed by utterance id
  vectors = np.array(vectors)
  pca = PCA(n_components=output_dim)
  low_dim_vectors = pca.fit_transform(vectors)
  logging.info(
      "The variance explained ratio for each dim of the dim-reduced vectors is {}".format(pca.explained_variance_ratio_))
  with VectorWriter(vector_wspecifier) as vector_writer:
    for i, vector in enumerate(low_dim_vectors):
      vector_writer[uttids[i]] = vector
  return True


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Use Principal component analysis for dimension reduction.
  For the details, Please refer to website:
  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

  Usage: pca-vector.py [options] <vector-rspecifier> <vector-wspecifier

  e.g.
      pca-vector.py scp:data/train/ivector.scp ark:data/train/low_dim_vector.ark

  see also: two-dim-vector-visual.py
  """
  po = ParseOptions(usage)
  po.register_int("output-dim", 2,
                  "dimension of the output vectors."
                  " For visualization, only 2 is allowed in this program. (2 by default)")
  opts = po.parse_args()
  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  vector_rspecifier = po.get_arg(1)
  vector_wspecifier = po.get_arg(2)
  isSuccess = pca_vector(vector_rspecifier,
                         vector_wspecifier,
                         output_dim=opts.output_dim)
  if not isSuccess:
    sys.exit()
