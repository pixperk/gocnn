package nn

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

func TestReLUForward(t *testing.T) {
	x := from([]float32{-2, 1.5, -0.5, 3, 0, -0.1}, 2, 3)
	y := (&ReLU{}).Forward(x)

	if !slices.Equal(y.Shape, []int{2, 3}) {
		t.Fatalf("Shape = %v, want [2 3]", y.Shape)
	}
	if want := []float32{0, 1.5, 0, 3, 0, 0}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
	if want := []float32{-2, 1.5, -0.5, 3, 0, -0.1}; !slices.Equal(x.Data, want) {
		t.Errorf("input changed: %v", x.Data)
	}
}

func TestReLUBackward(t *testing.T) {
	r := &ReLU{}
	r.Forward(from([]float32{-2, 1.5, -0.5, 3}, 1, 4))
	dx := r.Backward(from([]float32{0.4, 0.7, 0.1, 0.2}, 1, 4))

	if want := []float32{0, 0.7, 0, 0.2}; !slices.Equal(dx.Data, want) {
		t.Errorf("dx = %v, want %v", dx.Data, want)
	}
}

// at exactly zero the gate is closed.
func TestReLUBackwardAtZero(t *testing.T) {
	r := &ReLU{}
	r.Forward(from([]float32{0}, 1, 1))
	dx := r.Backward(from([]float32{5}, 1, 1))

	if dx.Data[0] != 0 {
		t.Errorf("dx at x=0 is %v, want 0", dx.Data[0])
	}
}

func TestReLUHasNoParams(t *testing.T) {
	if ps := (&ReLU{}).Params(); len(ps) != 0 {
		t.Errorf("Params() = %v, want none", ps)
	}
}

func TestReLUIsLayer(t *testing.T) {
	var _ Layer = &ReLU{}
}

// inputs stay at least 0.1 away from zero so the finite-difference nudge
// never crosses the kink.
func TestReLUGradCheck(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	x := tensor.New(4, 5)
	for i := range x.Data {
		v := 0.1 + r.Float32()*0.9
		if r.IntN(2) == 0 {
			v = -v
		}
		x.Data[i] = v
	}
	gradCheck(t, &ReLU{}, x, r)
}
