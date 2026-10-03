# gocnn

A convolutional neural network in Go, written from scratch. No PyTorch, no
gonum, no cgo: tensors, backpropagation, and training are built on the
standard library alone.

It is a learning project. The aim is to understand the machinery that ML
frameworks hide, then make it fast.

## Status

A two-layer network trains on MNIST and reads handwritten digits:

```
$ go run ./cmd/train
train 60000, test 10000, before training: test acc 8.42%
epoch 1  loss 0.3750  test acc 93.64%  (12.88s)
epoch 2  loss 0.2023  test acc 94.94%  (12.968s)
epoch 3  loss 0.1512  test acc 96.11%  (12.833s)
epoch 4  loss 0.1216  test acc 96.58%  (12.846s)
epoch 5  loss 0.1024  test acc 96.91%  (12.882s)
```

That is `Linear(784→128) → ReLU → Linear(128→10)` with plain SGD, measured on
the 10,000 test images it never trains on. The convolutional layers are next.

## Run it

Requires Go 1.27.

Download MNIST into `data/` (about 11 MB):

```sh
mkdir -p data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fsSL -o "data/$f.gz" "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz"
done
```

Train:

```sh
go run ./cmd/train                     # 5 epochs, batch 64, lr 0.1
go run ./cmd/train -epochs 10 -lr 0.05
```

| Flag | Default | Meaning |
|---|---|---|
| `-epochs` | `5` | Passes over the training set |
| `-batch` | `64` | Examples per step |
| `-lr` | `0.1` | Learning rate |
| `-seed` | `1` | Random seed; the same seed gives the same run |
| `-data` | `data` | Directory holding the `.gz` files |

Watch it learn in the browser, and draw your own digits for it to read:

```sh
go run ./cmd/demo                      # then open http://localhost:8080
```

Test:

```sh
go test ./...
```

## How it is built

| Package | What it does |
|---|---|
| [`tensor`](tensor) | N-dimensional `float32` tensors: strided views, reshape, transpose, broadcasting, elementwise ops, reductions, matmul |
| [`nn`](nn) | The `Layer` interface, `Linear`, `ReLU`, and softmax cross-entropy loss, each with a hand-written backward pass |
| [`optim`](optim) | Stochastic gradient descent |
| [`dataset`](dataset) | MNIST IDX reader and batching |
| [`cmd/train`](cmd/train) | The training loop |
| [`cmd/demo`](cmd/demo) | A live training dashboard with a drawing pad, served from one Go binary |

A few ideas carry most of the weight:

- **A tensor is a flat slice plus shape and strides.** Transpose swaps two
  strides and reshape recomputes them, so neither copies data.
- **Broadcasting is stride 0.** Adding a bias of shape `[10]` to a batch of
  shape `[64, 10]` reads the same ten numbers for every row, without copying
  them.
- **One odometer walks any tensor.** A single loop steps through every index
  of any shape and tracks each tensor's memory offset from its own strides.
  Elementwise ops, reductions, and contiguous copies all use it.
- **Reductions are broadcasting in reverse.** The output is expanded over the
  reduced axis with stride 0, so every value along that axis lands in the same
  slot.
- **Every backward pass is gradient-checked.** Tests nudge each weight and
  input by a small amount, measure how the loss really moves, and compare that
  with what `Backward` computed.

## What's ahead

- [x] **Tensor engine**: strides, views, broadcasting, reductions, matmul
- [x] **First network**: Linear, ReLU, softmax cross-entropy, SGD, MNIST training
- [ ] **Convolutions**: Flatten, MaxPool, Conv2D, then a small CNN that should
  reach about 99%
- [ ] **Make it fast**: benchmark and profile, then loop reordering,
  goroutines, cache tiling, and im2col convolution, each step measured against
  the 12.9 s/epoch baseline above
- [ ] **Autograd**: a reverse-mode engine, tested against the hand-written
  backward passes
- [ ] **Metal backend**: GPU kernels on Apple Silicon behind a backend
  interface
- [ ] **Inference**: model files, int8 quantisation, and an HTTP server that
  reads digits
