package nn

import (
	"math"
	"math/rand/v2"

	"github.com/pixperk/gocnn/tensor"
)

// linear computes y = xW + b.
type Linear struct {
	W, B *Param
	x    *tensor.Tensor
}

// newlinear makes a layer mapping in features to out, he-initialised.
func NewLinear(in, out int, r *rand.Rand) *Linear {
	std := math.Sqrt(2 / float64(in))
	w := tensor.New(in, out)
	for i := range w.Data {
		w.Data[i] = float32(r.NormFloat64() * std)
	}
	return &Linear{W: newParam(w), B: newParam(tensor.New(out))}
}

// forward caches x for backward and returns xW + b.
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	l.x = x
	return tensor.Add(tensor.MatMul(x, l.W.Value), l.B.Value)
}

func (l *Linear) Backward(dy *tensor.Tensor) *tensor.Tensor {
	dW := tensor.MatMul(l.x.Transpose(0, 1), dy)
	db := dy.Sum(0, false)
	dx := tensor.MatMul(dy, l.W.Value.Transpose(0, 1))
	//paraam update
	l.W.Grad = tensor.Add(l.W.Grad, dW)
	l.B.Grad = tensor.Add(l.B.Grad, db)
	return dx
}

// params returns w and b.
func (l *Linear) Params() []*Param {
	return []*Param{l.W, l.B}
}
