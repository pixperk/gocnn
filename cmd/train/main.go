package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"time"

	"github.com/pixperk/gocnn/dataset"
	"github.com/pixperk/gocnn/models"
	"github.com/pixperk/gocnn/nn"
	"github.com/pixperk/gocnn/optim"
)

func main() {
	dataDir := flag.String("data", "data", "directory holding the mnist .gz files")
	epochs := flag.Int("epochs", 5, "passes over the training set")
	batchSize := flag.Int("batch", 64, "examples per step")
	lr := flag.Float64("lr", 0.1, "learning rate")
	seed := flag.Uint64("seed", 1, "random seed")
	arch := flag.String("model", "mlp", "model to train: "+strings.Join(models.Names(), ", "))
	flag.Parse()

	train, err := dataset.Load(
		filepath.Join(*dataDir, "train-images-idx3-ubyte.gz"),
		filepath.Join(*dataDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		log.Fatal(err)
	}
	test, err := dataset.Load(
		filepath.Join(*dataDir, "t10k-images-idx3-ubyte.gz"),
		filepath.Join(*dataDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		log.Fatal(err)
	}

	r := rand.New(rand.NewPCG(*seed, *seed))
	model, err := models.Build(*arch, r)
	if err != nil {
		log.Fatal(err)
	}
	params := model.Params()
	var lossFn nn.SoftmaxCrossEntropy
	opt := &optim.SGD{LR: float32(*lr)}

	fmt.Printf("%s: %s\ntrain %d, test %d, before training: test acc %.2f%%\n",
		*arch, models.Label(*arch), train.Len(), test.Len(), accuracy(model, test))

	for epoch := 1; epoch <= *epochs; epoch++ {
		start := time.Now()
		perm := r.Perm(train.Len())

		var total float64
		var steps int
		for s := 0; s < len(perm); s += *batchSize {
			x, y := train.Batch(perm[s:min(s+*batchSize, len(perm))])

			loss := lossFn.Forward(model.Forward(x), y)
			model.Backward(lossFn.Backward())
			opt.Step(params)

			total += float64(loss)
			steps++
			if epoch == 1 && (steps == 1 || steps%100 == 0) {
				fmt.Printf("  step %4d  loss %.4f\n", steps, loss)
			}
		}

		fmt.Printf("epoch %d  loss %.4f  test acc %.2f%%  (%s)\n",
			epoch, total/float64(steps), accuracy(model, test), time.Since(start).Round(time.Millisecond))
	}
}

// accuracy is the percentage of d the model labels correctly.
func accuracy(model nn.Layer, d *dataset.Dataset) float64 {
	const chunk = 1000
	correct := 0
	for s := 0; s < d.Len(); s += chunk {
		idx := make([]int, 0, chunk)
		for i := s; i < min(s+chunk, d.Len()); i++ {
			idx = append(idx, i)
		}
		x, y := d.Batch(idx)
		for i, p := range model.Forward(x).ArgMax(1) {
			if p == y[i] {
				correct++
			}
		}
	}
	return 100 * float64(correct) / float64(d.Len())
}
