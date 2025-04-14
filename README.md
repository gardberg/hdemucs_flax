# HDemucs Flax

Flax implementation of HDemucs audio source separation model

## TODO

Forward pass is working, output sounds ok! small numerical error though, unsure of why.
- [ ] Subclass TorchConv, TorchConv2d, and GroupNorm to add logging logic for debugging?

- [x] HDemucs
  - [ ] Utility tests
  - [x] e2e-tests
  - [x] Numerical diff test
  - [ ] JIT-compilation tests
  - [ ] Speed comparison
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
  - How do we handle modules that do not subclass the custom module?

## Notes

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

