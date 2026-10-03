package nn

import (
	"slices"

	"github.com/pixperk/gocnn/tensor"
)

// flatten squashes every dim after the batch into one.
type Flatten struct {
	shape []int
}

// forward reshapes x to (n, -1), remembering its shape.
func (f *Flatten) Forward(x *tensor.Tensor) *tensor.Tensor {
	f.shape = slices.Clone(x.Shape)
	return x.Contiguous().Reshape(x.Shape[0], -1)
}

// backward reshapes dy back to the input's shape.
func (f *Flatten) Backward(dy *tensor.Tensor) *tensor.Tensor {
	return dy.Contiguous().Reshape(f.shape...)
}

// params returns nothing; flatten has no weights.
func (f *Flatten) Params() []*Param {
	return nil
}
