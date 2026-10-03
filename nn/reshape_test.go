package nn

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

func TestReshapeLayer(t *testing.T) {
	r := NewReshape(1, 2, 2)
	x := from([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)

	y := r.Forward(x)
	if !slices.Equal(y.Shape, []int{2, 1, 2, 2}) {
		t.Fatalf("Shape = %v, want [2 1 2 2]", y.Shape)
	}
	if !slices.Equal(y.Data, x.Data) {
		t.Errorf("values changed: %v", y.Data)
	}

	dx := r.Backward(tensor.New(2, 1, 2, 2))
	if !slices.Equal(dx.Shape, []int{2, 4}) {
		t.Errorf("dx shape = %v, want [2 4]", dx.Shape)
	}
}

func TestReshapeLayerKeepsBatch(t *testing.T) {
	y := NewReshape(1, 28, 28).Forward(tensor.New(5, 784))
	if !slices.Equal(y.Shape, []int{5, 1, 28, 28}) {
		t.Errorf("Shape = %v, want [5 1 28 28]", y.Shape)
	}
}

func TestReshapeLayerOwnsShape(t *testing.T) {
	shape := []int{1, 2, 2}
	r := NewReshape(shape...)
	shape[0] = 99

	if y := r.Forward(tensor.New(1, 4)); !slices.Equal(y.Shape, []int{1, 1, 2, 2}) {
		t.Errorf("Shape = %v, want [1 1 2 2]", y.Shape)
	}
}

func TestReshapeLayerIsLayer(t *testing.T) {
	var _ Layer = NewReshape(1)
	if ps := NewReshape(1).Params(); len(ps) != 0 {
		t.Errorf("Params() = %v, want none", ps)
	}
}

func TestReshapeLayerGradCheck(t *testing.T) {
	r := rand.New(rand.NewPCG(15, 16))
	x := tensor.New(3, 8)
	for i := range x.Data {
		x.Data[i] = r.Float32()*2 - 1
	}
	gradCheck(t, NewReshape(2, 2, 2), x, r)
}
