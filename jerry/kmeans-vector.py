#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2020 Jerry Peng

# This script use k-means to assign cluster label for vectors.

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialVectorReader, VectorWriter, read_script_file,\
    classify_rspecifier, RspecifierType,\
    classify_wspecifier, WspecifierType
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pdb


def kmeans_vector(vector_rspecifier, utt2clusterid_rxfilename, n_clusters=2, random_state=0, batch_size=6, max_iter=100):
  vectors = []
  uttids = []
  with SequentialVectorReader(vector_rspecifier) as vector_reader:
    for uttid, vector in vector_reader:
      uttids.append(uttid)
      vectors.append(vector.numpy())
  # vectors is a set of row vectors indexed by utterance id
  vectors = np.array(vectors)
  mix_labels = MiniBatchKMeans(n_clusters=n_clusters,
                               random_state=random_state,
                               batch_size=batch_size,
                               max_iter=max_iter).fit_predict(vectors)
  # write mix_labels to utt2clusterid_rxfilename file
  with open(utt2clusterid_rxfilename, 'w') as fout:
    for i in range(len(uttids)):
      fout.write("{uttid} cluster{mixid}\n".format(
          uttid=uttids[i], mixid=mix_labels[i]))
  return True


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Use MiniBatchKMeans for vector clustering. It outputs cluster assignments
  For the details, Please refer to website:
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans

  Usage: kmeans-vector.py [options] <vector-rspecifier> <utt2clusterid-rxfilename>

  e.g.
      kmeans-vector.py scp:data/train/ivector.scp data/train/utt2clusterid
  """
  po = ParseOptions(usage)
  po.register_int("n-clusters", 8,
                  "The number of clusters to form as well as the number of centroids to generate. default=8")
  po.register_int("random-state", 0,
                  "Determines random number generation for centroid initialization and random reassignment. "
                  "Use an int to make the randomness deterministic. ")
  po.register_int("batch-size", 6,
                  "Size of the mini batches.")
  po.register_int("max-iter", 100,
                  "Maximum number of iterations over the complete dataset before stopping independently of "
                  "any early stopping criterion heuristics.")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  vector_rspecifier = po.get_arg(1)
  utt2clusterid_rxfilename = po.get_arg(2)
  isSuccess = kmeans_vector(vector_rspecifier,
                            utt2clusterid_rxfilename,
                            n_clusters=opts.n_clusters,
                            random_state=opts.random_state,
                            batch_size=opts.batch_size,
                            max_iter=opts.max_iter)
  if not isSuccess:
    sys.exit()
