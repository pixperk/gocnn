package nn

import "github.com/pixperk/gocnn/tensor"

// relu computes max(0, x) elementwise.
type ReLU struct {
	mask *tensor.Tensor
}

// forward stores which inputs were positive and returns max(0, x).
func (r *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	r.mask = x.Apply(func(v float32) float32 {
		if v > 0 {
			return 1
		}
		return 0
	})

	return x.Apply(func(v float32) float32 {
		return max(0, v)
	})
}

// backward passes dy through where the input was positive.
func (r *ReLU) Backward(dy *tensor.Tensor) *tensor.Tensor {
	return tensor.Mul(dy, r.mask)
}

// params returns nothing; relu has no weights.
func (r *ReLU) Params() []*Param {
	return nil
}
