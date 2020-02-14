#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2020 Jerry Peng

# format kaldi data into json file. use it for website display

from __future__ import print_function, division

import sys
import logging
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialVectorReader, VectorWriter, read_script_file,\
                            classify_rspecifier, RspecifierType,\
                            classify_wspecifier, WspecifierType
import numpy as np
import json
import pdb


def to_json(name, vector2d_rspecifier, utt2spk_rxfilename, utt2wav_rxfilename, json_rxfilename):
  all_json = {}
  wav2path = {}
  utt2spk = {}
  vectors = []
  uttids = []
  paths = []
  spkids = []
  with open(utt2wav_rxfilename, 'r') as fi:
    for line in fi:
      uttid, path = line.rstrip().split(' ')
      wav2path[uttid] = path
  with open(utt2spk_rxfilename, 'r') as fi:
    for line in fi:
      uttid, spkid = line.rstrip().split(' ')
      utt2spk[uttid] = spkid

  with SequentialVectorReader(vector_rspecifier) as vector_reader:
    for uttid, vector in vector_reader:
      vectors.append(vector.numpy().tolist())
      uttids.append(uttid)
      paths.append(wav2path[uttid])
      spkids.append(utt2spk[uttid])

  all_json[str(name)] = vectors
  all_json['uttids'] = uttids
  all_json['paths'] = paths
  all_json['spkids'] = spkids
  # write json
  json_string = "points = '" + json.dumps(all_json) + "'"
  with open(json_rxfilename, 'w') as json_file:
    json_file.write(json_string)
  return True


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Format data into json format. use it for website 2D display

  Usage: to-json.py [options] <json-var-name> <two-dim-vector-rspecifier> <utt2spk-rxfilename> <utt2wav-rxfilename> <json-rxfilename>

  e.g.
      to-json.py --json-var-name=mfvae-tsne scp:data/train/ivector2d.scp data/train/utt2spk data/train/wav.scp exp/mfvae_cmvn/data.json
  """
  po = ParseOptions(usage)
  po.register_str("json-var-name", "json-data",
                  "The variable name in json file. It will be used in javascript. default=\"json-data\"")
  opts = po.parse_args()

  if (po.num_args() != 4):
    po.print_usage()
    sys.exit()

  vector_rspecifier = po.get_arg(1)
  utt2spk_rxfilename = po.get_arg(2)
  utt2wav_rxfilename = po.get_arg(3)
  json_rxfilename = po.get_arg(4)
  isSuccess = to_json(opts.json_var_name,
                      vector_rspecifier,
                      utt2spk_rxfilename,
                      utt2wav_rxfilename,
                      json_rxfilename)
  if not isSuccess:
    sys.exit()

