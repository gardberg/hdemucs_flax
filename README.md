# HDemucs Flax

Flax implementation of HDemucs audio source separation model

## Dev

Install only base deps (needed for checkpoint compilation and inference):

`uv sync --no-dev`

Test build the package: `uv build`

Use package in another repo:

`uv add git+https://github.com/gardberg/hdemucs_flax.git`

This installs all the modules at root level! So use them via:

`from separate import Separator` for example.

## Tests

`pytest -k 'not speed' --durations=0 -v`

`pytest --benchmark-skip` / `pytest --benchmark-only`

## TODO

- [x] Switch to regular nnx.GroupNorm instead of TorchGroupNorm
- [x] Rewrite TorchConv, TorchConv2d with Module
- [x] Rewrite transposed convs to be jit compilable
- [x] Write logic for comparing numerical diffs (torch_hook.ipynb)

- [x] HDemucs
  - [x] Utility tests
  - [x] e2e-tests
  - [x] Numerical diff test
  - [x] JIT-compilation tests
  - [x] Speed comparison
- [x] ScaledEmbedding
- [x] _HencLayer
  - [x] Time-tests
  - [x] _DConv
    - [x] _LayerScale
    - [x] _BLSTM
      - [x] BidirectionalLSTM
  - [x] _LocalState
- [x] _HDecLayer
  - [x] Freq tests
  - [x] Time tests

- [ ] Ruff linting
- [ ] Pyright type checking
- [ ] Shape annotations
- [x] Custom interceptor logic
- [ ] Dockerized inference API
- [ ] CI tests

## Notes

Currently need to decide how to handle BLSTM framing logic for compiling with lengths
over ~4 seconds of input. Can probably set fixed padding, but how do we handle 
discarding extra length outside of blstm? it is also dependent on input shape

#### Speed considerations

- Approximate GELU / GLU?
- Fast variance via flax groupnorm (fused kernel)?
- Export / serialize and run via IREE-compiled model?

#### Api

To build API image, run:

`docker build -t hdemucs_api .`


#### Misc

Model used is HDEMUCS_HIGH_MUSDB_PLUS from torchaudio

https://flax-linen.readthedocs.io/en/latest/guides/converting_and_upgrading/convert_pytorch_to_flax.html

- freq_encoder: 6x _HEncLayer
- freq_decoder: 6x _HDecLayer
- time_encoder: 5x _HEncLayer
- time_decoder: 5x _HDecLayer
- freq_emb    : 1x _ScaledEmbedding

Using HDemucs high (44.1 - 48 kHz)

nfft: 4096
depth = 6

Rest default params

![HDemucs Architecture](./images/arch.png)

