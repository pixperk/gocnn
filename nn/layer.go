package nn

import "github.com/pixperk/gocnn/tensor"

// param is a learnable tensor and the gradient of the loss with respect to it.
type Param struct {
	Value *tensor.Tensor
	Grad  *tensor.Tensor
}

// newParam wraps value with a zero gradient of the same shape.
func newParam(value *tensor.Tensor) *Param {
	return &Param{Value: value, Grad: tensor.New(value.Shape...)}
}

// layer is one step of the network: forward predicts, backward passes gradients back.
type Layer interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Backward(dy *tensor.Tensor) *tensor.Tensor
	Params() []*Param
}
