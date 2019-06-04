# tfcheckpoint2pytorch
This converter can convert TensorFlow checkpoints (directories and tarballs containing an `*.index`, `*.meta` and `*.data-*-of-*` files).

For example, let's download NVidia OpenSeq2Seq [wav2letter model checkpoint](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/wave2letter.html#training) and dump the weights. The file name will be `./w2l_plus_large.tar.gz` (this archive contains index, meta and data files).
```
# convert the checkpoint to PyTorch
python3 tfcheckpoint2pytorch.py --checkpoint ./w2l_plus_large.tar.gz -o ./w2l_plus_large.pt

# convert the checkpoint to HDF5
python3 tfcheckpoint2pytorch.py --checkpoint ./w2l_plus_large.tar.gz -o ./w2l_plus_large.h5
h5ls ./w2l_plus_large.h5
```

**Dependencies:** Unforutanately this converter requires TensorFlow installed. However, it's okay even if it's installed via pip: `pip3 install tensorflow`. PyTorch and h5py are optional dependencies.
