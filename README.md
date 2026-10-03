# gocnn

A convolutional neural network in Go, written from scratch. No PyTorch, no
gonum, no cgo: tensors, backpropagation, and training are built on the
standard library alone.

It is a learning project. The aim is to understand the machinery that ML
frameworks hide, then make it fast.

## Status

A small convolutional network reads handwritten digits at 98.4% accuracy:

```
$ go run ./cmd/train -model cnn
cnn: Conv 1→8 · Pool · Conv 8→16 · Pool · Linear 784→10
train 60000, test 10000, before training: test acc 13.11%
epoch 1  loss 0.2151  test acc 97.27%  (41.827s)
epoch 2  loss 0.0799  test acc 98.01%  (42.021s)
epoch 3  loss 0.0629  test acc 98.29%  (41.791s)
epoch 4  loss 0.0541  test acc 98.36%  (41.837s)
epoch 5  loss 0.0488  test acc 98.05%  (42.366s)
```

Accuracy is measured on the 10,000 test images the network never trains on.
Both models train with plain SGD:

| Model | Layers | Parameters | Best test accuracy (5 epochs) | Time per epoch |
|---|---|---|---|---|
| `mlp` | Linear 784→128 · ReLU · Linear 128→10 | 101,770 | 96.91% | 12.9 s |
| `cnn` | Conv 1→8 · ReLU · Pool · Conv 8→16 · ReLU · Pool · Linear 784→10 | 9,098 | 98.36% | 41.8 s |

The CNN is more accurate with a tenth of the parameters, because each 3×3
filter is reused at every position in the image. It is also three times
slower, because convolution is still written as plain nested loops.

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
go run ./cmd/train                     # mlp, 5 epochs, batch 64, lr 0.1
go run ./cmd/train -model cnn          # the convolutional network
```

| Flag | Default | Meaning |
|---|---|---|
| `-model` | `mlp` | `mlp` or `cnn` |
| `-epochs` | `5` | Passes over the training set |
| `-batch` | `64` | Examples per step |
| `-lr` | `0.1` | Learning rate |
| `-seed` | `1` | Random seed; the same seed gives the same run |
| `-data` | `data` | Directory holding the `.gz` files |

Watch it learn in the browser, and draw your own digits for it to read:

```sh
go run ./cmd/demo -model cnn           # then open http://localhost:8080
```

Test:

```sh
go test ./...
```

## How it is built

| Package | What it does |
|---|---|
| [`tensor`](tensor) | N-dimensional `float32` tensors: strided views, reshape, transpose, broadcasting, elementwise ops, reductions, matmul |
| [`nn`](nn) | The `Layer` interface and layers, each with a hand-written backward pass: `Linear`, `Conv2D`, `MaxPool2D`, `ReLU`, `Flatten`, `Reshape`, plus softmax cross-entropy loss |
| [`models`](models) | The `mlp` and `cnn` networks, shared by training and the demo |
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
- [x] **Convolutions**: Conv2D, MaxPool2D, Flatten, and a CNN at 98.4%
- [ ] **Make it fast**: benchmark and profile, then loop reordering,
  goroutines, cache tiling, and im2col convolution, each step measured against
  the 12.9 s and 41.8 s per-epoch baselines above
- [ ] **Autograd**: a reverse-mode engine, tested against the hand-written
  backward passes
- [ ] **Metal backend**: GPU kernels on Apple Silicon behind a backend
  interface
- [ ] **Inference**: model files, int8 quantisation, and an HTTP server that
  reads digits
