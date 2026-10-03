package nn

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

func TestSequentialParams(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	a, b := NewLinear(3, 4, r), NewLinear(4, 2, r)
	s := NewSequential(a, &ReLU{}, b)

	want := []*Param{a.W, a.B, b.W, b.B}
	if got := s.Params(); !slices.Equal(got, want) {
		t.Errorf("Params() = %v, want %v", got, want)
	}
}

func TestSequentialForwardMatchesLayers(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	a, relu, b := NewLinear(3, 4, r), &ReLU{}, NewLinear(4, 2, r)
	x := from([]float32{1, -2, 3, 0.5, 0.5, -1}, 2, 3)

	want := b.Forward(relu.Forward(a.Forward(x)))
	got := NewSequential(a, relu, b).Forward(x)

	if !slices.Equal(got.Data, want.Data) {
		t.Errorf("Forward = %v, want %v", got.Data, want.Data)
	}
}

func TestSequentialGradCheck(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 9))
	s := NewSequential(NewLinear(5, 6, r), &ReLU{}, NewLinear(6, 3, r))
	x := tensor.New(4, 5)
	for i := range x.Data {
		x.Data[i] = r.Float32()*2 - 1
	}
	gradCheck(t, s, x, r)
}
