package nn

import "github.com/pixperk/gocnn/tensor"

// sequential runs its layers in order and is itself a layer.
type Sequential struct {
	Layers []Layer
}

// newsequential chains layers into one model.
func NewSequential(layers ...Layer) *Sequential {
	return &Sequential{Layers: layers}
}

// forward runs x through every layer in order.
func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

// backward runs dy through every layer in reverse.
func (s *Sequential) Backward(dy *tensor.Tensor) *tensor.Tensor {
	for i := len(s.Layers) - 1; i >= 0; i-- {
		dy = s.Layers[i].Backward(dy)
	}
	return dy
}

// params returns every layer's params in order.
func (s *Sequential) Params() []*Param {
	var ps []*Param
	for _, l := range s.Layers {
		ps = append(ps, l.Params()...)
	}
	return ps
}
