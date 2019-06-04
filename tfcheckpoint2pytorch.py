import os
import sys
import argparse
import tempfile
import tarfile

import numpy as np
import tensorflow
from tensorflow.python import pywrap_tensorflow

parser = argparse.ArgumentParser()
parser.add_argument('--tmp_dir', default = tempfile.mkdtemp())
parser.add_argument('-i', '--checkpoint')
parser.add_argument('-o', '--output_path')
args = parser.parse_args()

if args.checkpoint.endswith('.tar.gz') or args.checkpoint.endswith('.tar'):
    checkpoint_dir = args.tmp_dir
    tarfile.open(args.checkpoint).extractall(checkpoint_dir)
    files = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)]
    checkpoint_dir = files[0] if os.path.isdir(files[0]) else checkpoint_dir
else:
    checkpoint_dir = args.checkpoint

reader = pywrap_tensorflow.NewCheckpointReader(tensorflow.train.latest_checkpoint(checkpoint_dir))
blobs = {k : reader.get_tensor(k) for k in reader.get_variable_to_shape_map()}

if args.output_path.endswith('.json'):
    import json
    with open(args.output_path, 'w') as f:
        json.dump({k : blob.tolist() for k, blob in blobs.items()}, f, sort_keys = True, indent = 2)
elif args.output_path.endswith('.h5'):
    import h5py
    with h5py.File(args.output_path, 'w') as h:
        h.update(**blobs)
elif args.output_path.endswith('.npy') or args.output_path.endswith('.npz'):
    (np.savez if args.output_path[-1] == 'z' else numpy.save)(args.output_path, **blobs)
elif args.output_path.endswith('.pt'):
    import torch
    torch.save({k : torch.from_numpy(blob) for k, blob in blobs.items()}, args.output_path)
