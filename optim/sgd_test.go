package optim

import (
	"slices"
	"testing"

	"github.com/pixperk/gocnn/nn"
	"github.com/pixperk/gocnn/tensor"
)

func param(value, grad []float32) *nn.Param {
	p := &nn.Param{Value: tensor.New(len(value)), Grad: tensor.New(len(grad))}
	copy(p.Value.Data, value)
	copy(p.Grad.Data, grad)
	return p
}

func TestSGDStep(t *testing.T) {
	a := param([]float32{1, 2, 3}, []float32{10, -10, 0})
	b := param([]float32{5}, []float32{2})

	(&SGD{LR: 0.5}).Step([]*nn.Param{a, b})

	if want := []float32{-4, 7, 3}; !slices.Equal(a.Value.Data, want) {
		t.Errorf("a = %v, want %v", a.Value.Data, want)
	}
	if want := []float32{4}; !slices.Equal(b.Value.Data, want) {
		t.Errorf("b = %v, want %v", b.Value.Data, want)
	}
}

func TestSGDZeroesGrads(t *testing.T) {
	p := param([]float32{1, 2}, []float32{3, 4})

	(&SGD{LR: 0.1}).Step([]*nn.Param{p})

	if want := []float32{0, 0}; !slices.Equal(p.Grad.Data, want) {
		t.Errorf("grad after step = %v, want %v", p.Grad.Data, want)
	}
}
