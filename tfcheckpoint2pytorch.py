import os
import sys
import glob
import json
import argparse
import tempfile
import shutil
import tarfile
import collections

import numpy as np
import tensorflow
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import meta_graph
from tensorflow.core.framework import types_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--tmp_dir', default = tempfile.mkdtemp())
parser.add_argument('-i', '--checkpoint')
parser.add_argument('-o', '--output_path', default = '')
parser.add_argument('--onnx')
parser.add_argument('--tensorboard')
parser.add_argument('--graph')
parser.add_argument('--identityop', action = 'append', default = [])
parser.add_argument('--ignoreattr', action = 'append', default = [])
parser.add_argument('--input_name', action = 'append', default = [])
parser.add_argument('--input_shape', action = 'append', nargs = '+', type = int)
parser.add_argument('--input_dtype', action = 'append', type = str)
parser.add_argument('--output_name', action = 'append', default = [])
parser.add_argument('--opset', default = 10, type = int)
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

if args.onnx or args.tensorboard or args.graph:
	meta_graph_file = glob.glob(os.path.join(checkpoint_dir, '*.meta'))[0]
	graph_def = meta_graph.read_meta_graph_file(meta_graph_file).graph_def

	if args.graph or (not args.input_name) or (args.onnx and not args.output_name):
		print('\n'.join(sorted(f'{v.name} <- {node.op}(' + ', '.join(v.input) for v in graph_def.node)) + ')', file = None if not args.graph else open(args.graph, 'w'))
		sys.exit(0)

	for v in graph_def.node:
		if any(name in v.name for name in args.identityop):
			v.op = 'Identity'
			for a in set(v.attr.keys()) - set(['T']):
				del v.attr[a]

	with_port_id = lambda n: n if ':' in n else n + ':0'
	without_port_id = lambda n: n.split(':')[0]
	port_id = lambda n: int(n.split(':')[1]) if ':' in n else 0

	def input_type_shape(n):
		port = port_id(n)
		name = without_port_id(n)
		i = args.input_name.index(n)
		nodes = [v for v in graph_def.node if name in v.name]

		if args.input_shape:
			shape = [None if d == -1 else d for d in args.input_shape[i]]
		else:
			shapes = nodes[0].attr['output_shapes'].list.shape
			shape = [None if d.size == -1 else d.size for d in shapes[port].dim]

		if args.input_dtype:
			dtype = getattr(tensorflow, args.input_dtype[i])
		else:
			dtype = nodes[0].attr['output_types'].list.type[port]

		return dtype, shape

	input_map = {with_port_id(a) : tensorflow.placeholder(*input_type_shape(a), name = without_port_id(a)) for a in args.input_name}

	tensorflow.import_graph_def(graph_def, input_map = input_map)
	graph = tensorflow.get_default_graph()

if args.tensorboard:
	if os.path.exists(args.tensorboard):
		shutil.rmtree(args.tensorboard)
	tensorflow.summary.FileWriter(args.tensorboard, graph = graph).close()
	
if args.onnx:
	import onnx, tf2onnx

	tf2onnx.utils.TF_TO_ONNX_DTYPE[types_pb2.DT_VARIANT] = tf2onnx.utils.TF_TO_ONNX_DTYPE[types_pb2.DT_FLOAT]
	for t in dir(types_pb2):
		if t.endswith('_REF'):
			tf2onnx.utils.TF_TO_ONNX_DTYPE[getattr(types_pb2, t)] = tf2onnx.utils.TF_TO_ONNX_DTYPE.get(getattr(types_pb2, t[:-len('_REF')], None))
	onnx.helper.make_node = lambda *args_, make_node_ = onnx.helper.make_node, **kwargs_: make_node_(*args_, **{k: v for k, v in kwargs_.items() if k not in args.ignoreattr})

	input_names = [with_port_id(a.name) for a in input_map.values()]
	output_names = ['import/' + with_port_id(o) for o in args.output_name]
	onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph, input_names = input_names, output_names = output_names, continue_on_error = True, opset = args.opset)
	model_proto = onnx_graph.make_model(os.path.basename(args.onnx))
	with open(args.onnx, "wb") as f:
		f.write(model_proto.SerializeToString())
