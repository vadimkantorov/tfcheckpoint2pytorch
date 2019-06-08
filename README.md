# tfcheckpoint2pytorch
This tool can dump model weights from TensorFlow checkpoints (directories and tarballs containing an `*.index`, `*.meta` and `*.data-*-of-*` files) to PyTorch, HDF5 NumPy binary formats and to JSON. It also can use [tf2onnx](https://github.com/onnx/tensorflow-onnx) and try to export the model to the ONNX format.

For example, let's download NVidia OpenSeq2Seq [wav2letter model checkpoint](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html#training) and dump the weights. The file name will be `./w2l_plus_large.tar.gz` (this archive contains index, meta and data files).

# Dump weights to PyTorch binary format and to HDF5
```
# convert the checkpoint to PyTorch
python3 tfcheckpoint2pytorch.py --checkpoint ./w2l_plus_large.tar.gz -o ./w2l_plus_large.pt

# convert the checkpoint to HDF5
python3 tfcheckpoint2pytorch.py --checkpoint ./w2l_plus_large.tar.gz -o ./w2l_plus_large.h5
h5ls ./w2l_plus_large.h5
```

**Dependencies:** Unforutanately this converter requires TensorFlow installed. However, it's okay even if it's installed via pip: `pip3 install tensorflow`. PyTorch and h5py are optional dependencies.

# Example: export openseq2seq's wav2letter speech2text model to ONNX format
We will try to export [NVidia openseq2seq's wav2letter speech2text model](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html) to ONNX. Unfortunately, tf2onnx [doesn't](https://github.com/onnx/tensorflow-onnx/issues/571) [support](https://github.com/onnx/tensorflow-onnx/issues/572) properly the BatchToSpaceND op that TensorFlow uses to implement dilated convolutions. So it doesn't work perfectly, but you can still probably use the result. Feel free to explore the produced `*.onnx` file in [Lutz Roeder's Netron online model explorer](https://lutzroeder.github.io/netron/).

```shell
CHECKPOINT_GOOGLE_DRIVE_URL='https://drive.google.com/file/d/10EYe040qVW6cfygSZz6HwGQDylahQNSa'
GOOGLE_DRIVE_FILE_ID=$(echo $CHECKPOINT_GOOGLE_DRIVE_URL | rev | cut -d'/' -f1 | rev)
CONFIRM=$(wget --quiet --save-cookies googlecookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$GOOGLE_DRIVE_FILE_ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies googlecookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$GOOGLE_DRIVE_FILE_ID" -O w2l_plus_large_mp.tar.gz || rm googlecookies.txt # from https://gist.github.com/vladalive/535cc2aff8a9527f1d9443b036320672

# export model weights to HDF5
python3 tfcheckpoint2pytorch.py --checkpoint w2l_plus_large_mp.tar.gz -o w2l_plus_large_mp.h5

# we must replace Horovod-related nodes by Identity, otherwise TensorFlow can't load the checkpoint
# https://github.com/horovod/horovod/issues/594

# print all variable names to help you identify input and output names
python3 tfcheckpoint2pytorch.py --checkpoint w2l_plus_large_mp.tar.gz --onnx w2l_plus_large_mp.onnx \
    --identity Horovod > graph.txt
    
# we must force tf2onnx and ONNX to ignore some node attributes:
# https://github.com/onnx/tensorflow-onnx/issues/578
# https://github.com/onnx/onnx/issues/2090
    
# export model to ONNX. you must specify input and output variable names, tfcheckpoint2pytorch will try to infer input shapes and dtype
python3 tfcheckpoint2pytorch.py --checkpoint w2l_plus_large_mp.tar.gz --onnx w2l_plus_large_mp.onnx \
    --identity Horovod \
    --ignoreattr Toutput_types --ignoreattr output_shapes --ignoreattr output_types --ignoreattr predicate --ignoreattr f --ignoreattr dtypes  \
    --output_name 'ForwardPass/fully_connected_ctc_decoder/logits:0' \
    --input_name 'IteratorGetNext:0'

# you can also force input shapes and dtype
python3 tfcheckpoint2pytorch.py --checkpoint w2l_plus_large_mp.tar.gz --onnx w2l_plus_large_mp.onnx \
    --identity Horovod \
    --ignoreattr Toutput_types --ignoreattr output_shapes --ignoreattr output_types --ignoreattr predicate --ignoreattr f --ignoreattr dtypes  \
    --input_name 'IteratorGetNext:0' --input_shape -1 -1 64 --input_dtype half \
    --output_name 'ForwardPass/fully_connected_ctc_decoder/logits:0'
     
```
