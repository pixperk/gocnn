package models

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"

	"github.com/pixperk/gocnn/nn"
)

type entry struct {
	label string
	build func(r *rand.Rand) *nn.Sequential
}

var registry = map[string]entry{
	"mlp": {
		label: "Linear 784→128 · ReLU · Linear 128→10",
		build: func(r *rand.Rand) *nn.Sequential {
			return nn.NewSequential(
				nn.NewLinear(784, 128, r),
				&nn.ReLU{},
				nn.NewLinear(128, 10, r),
			)
		},
	},
	"cnn": {
		label: "Conv 1→8 · Pool · Conv 8→16 · Pool · Linear 784→10",
		build: func(r *rand.Rand) *nn.Sequential {
			return nn.NewSequential(
				nn.NewReshape(1, 28, 28),
				nn.NewConv2D(1, 8, 3, 1, 1, r),
				&nn.ReLU{},
				nn.NewMaxPool2D(2),
				nn.NewConv2D(8, 16, 3, 1, 1, r),
				&nn.ReLU{},
				nn.NewMaxPool2D(2),
				&nn.Flatten{},
				nn.NewLinear(16*7*7, 10, r),
			)
		},
	},
}

// build makes the named model with weights drawn from r.
func Build(name string, r *rand.Rand) (*nn.Sequential, error) {
	e, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown model %q (have %s)", name, strings.Join(Names(), ", "))
	}
	return e.build(r), nil
}

// label is a short human description of the named model.
func Label(name string) string {
	return registry[name].label
}

// names lists the available models.
func Names() []string {
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	slices.Sort(names)
	return names
}
