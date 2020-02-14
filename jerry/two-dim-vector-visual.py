#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2020 Jerry Peng

# Given a set of 2-dimensional vectors, this script visualizes them.

from __future__ import print_function, division
import sys
import logging
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialVectorReader, VectorWriter, read_script_file
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # Added for runing on the server side
sns.set()


def two_dim_vector_visual(vector_rspecifier, utt2spk_rxfilename, figure_rxfilename):
  vectors = []
  spkidxs = []
  utt2spkidx = {}
  spk2idx = {}
  # load utt2spkidx, spk2idx
  with open(utt2spk_rxfilename, 'r') as fi:
    for line in fi:
      uttid, spkid = line.rstrip().split(' ')
      if spkid not in spk2idx:
        utt2spkidx[uttid] = len(spk2idx)
        spk2idx[spkid] = len(spk2idx)
      else:
        utt2spkidx[uttid] = spk2idx[spkid]
  # load spkidxs, vectors
  with SequentialVectorReader(vector_rspecifier) as vector_reader:
    for uttid, vector in vector_reader:
      spkidxs.append(utt2spkidx[uttid])
      vectors.append(vector.numpy())
  spkidxs = np.array(spkidxs)
  vectors = np.array(vectors)
  sns.scatterplot(vectors[:, 0], vectors[:, 1], spkidxs, palette=sns.color_palette("hls", len(spk2idx)))
  plt.savefig(figure_rxfilename)
  return True


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """save the visualization plot of 2-dimensional vectors to hardisk.

  Usage: two-dim-vector-visual.py [options] <vector-rspecifier> <utt2spk-rxfilename> <figure-rxfilename>

  e.g.
      two-dim-vector-visual.py scp:data/train/2d_vectors.scp data/train/utt2spk data/train/2d_vectors.png
  """
  po = ParseOptions(usage)
  opts = po.parse_args()

  if (po.num_args() != 3):
    po.print_usage()
    sys.exit()

  vector_rspecifier = po.get_arg(1)
  utt2spk_rxfilename = po.get_arg(2)
  figure_rxfilename = po.get_arg(3)
  isSuccess = two_dim_vector_visual(vector_rspecifier,
                                    utt2spk_rxfilename,
                                    figure_rxfilename)
  if not isSuccess:
    sys.exit()
