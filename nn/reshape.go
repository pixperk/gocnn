package nn

import (
	"slices"

	"github.com/pixperk/gocnn/tensor"
)

// reshape gives each example in the batch a new shape, e.g. 784 to (1, 28, 28).
type Reshape struct {
	Shape []int
	in    []int
}

// newreshape reshapes every example to shape.
func NewReshape(shape ...int) *Reshape {
	return &Reshape{Shape: slices.Clone(shape)}
}

// forward keeps the batch dim and reshapes the rest.
func (r *Reshape) Forward(x *tensor.Tensor) *tensor.Tensor {
	r.in = slices.Clone(x.Shape)
	return x.Contiguous().Reshape(append([]int{x.Shape[0]}, r.Shape...)...)
}

// backward reshapes dy back to the input's shape.
func (r *Reshape) Backward(dy *tensor.Tensor) *tensor.Tensor {
	return dy.Contiguous().Reshape(r.in...)
}

// params returns nothing; reshape has no weights.
func (r *Reshape) Params() []*Param {
	return nil
}
