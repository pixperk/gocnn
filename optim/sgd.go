package optim

import "github.com/pixperk/gocnn/nn"

// sgd moves every param a step of size lr against its gradient.
type SGD struct {
	LR float32
}

// step updates each param by -lr * grad, then zeroes the grad.
func (s *SGD) Step(params []*nn.Param) {
	for _, p := range params {
		for i, g := range p.Grad.Data {
			p.Value.Data[i] -= s.LR * g
		}
		clear(p.Grad.Data)
	}
}
