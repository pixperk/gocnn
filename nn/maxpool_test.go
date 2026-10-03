package nn

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

func TestMaxPoolForwardByHand(t *testing.T) {
	x := from([]float32{
		1, 3, 2, 1,
		6, 2, 0, 4,
		5, 8, 9, 1,
		0, 3, 2, 7,
	}, 1, 1, 4, 4)

	y := NewMaxPool2D(2).Forward(x)

	if !slices.Equal(y.Shape, []int{1, 1, 2, 2}) {
		t.Fatalf("Shape = %v, want [1 1 2 2]", y.Shape)
	}
	if want := []float32{6, 4, 8, 9}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

func TestMaxPoolBackwardByHand(t *testing.T) {
	m := NewMaxPool2D(2)
	m.Forward(from([]float32{
		1, 3, 2, 1,
		6, 2, 0, 4,
		5, 8, 9, 1,
		0, 3, 2, 7,
	}, 1, 1, 4, 4))

	dx := m.Backward(from([]float32{0.5, 0.1, 0.2, 0.7}, 1, 1, 2, 2))

	want := []float32{
		0, 0, 0, 0,
		0.5, 0, 0, 0.1,
		0, 0.2, 0.7, 0,
		0, 0, 0, 0,
	}
	if !slices.Equal(dx.Shape, []int{1, 1, 4, 4}) {
		t.Fatalf("dx shape = %v, want [1 1 4 4]", dx.Shape)
	}
	if !slices.Equal(dx.Data, want) {
		t.Errorf("dx = %v, want %v", dx.Data, want)
	}
}

// every image and channel is pooled on its own.
func TestMaxPoolChannelsAndBatch(t *testing.T) {
	x := from([]float32{
		1, 2, 3, 4, // n0 c0
		-1, -2, -3, -4, // n0 c1
		10, 0, 0, 0, // n1 c0
		0, 0, 0, 20, // n1 c1
	}, 2, 2, 2, 2)

	y := NewMaxPool2D(2).Forward(x)

	if !slices.Equal(y.Shape, []int{2, 2, 1, 1}) {
		t.Fatalf("Shape = %v, want [2 2 1 1]", y.Shape)
	}
	if want := []float32{4, -1, 10, 20}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

// a leftover row and column that do not fill a window are dropped.
func TestMaxPoolDropsRemainder(t *testing.T) {
	x := tensor.New(1, 1, 5, 5)
	for i := range x.Data {
		x.Data[i] = float32(i)
	}

	y := NewMaxPool2D(2).Forward(x)

	if !slices.Equal(y.Shape, []int{1, 1, 2, 2}) {
		t.Fatalf("Shape = %v, want [1 1 2 2]", y.Shape)
	}
	if want := []float32{6, 8, 16, 18}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

func TestMaxPoolTieGoesToFirst(t *testing.T) {
	m := NewMaxPool2D(2)
	m.Forward(from([]float32{5, 5, 5, 5}, 1, 1, 2, 2))
	dx := m.Backward(from([]float32{1}, 1, 1, 1, 1))

	if want := []float32{1, 0, 0, 0}; !slices.Equal(dx.Data, want) {
		t.Errorf("dx = %v, want %v", dx.Data, want)
	}
}

func TestMaxPoolMNISTShapes(t *testing.T) {
	y := NewMaxPool2D(2).Forward(tensor.New(4, 8, 28, 28))

	if !slices.Equal(y.Shape, []int{4, 8, 14, 14}) {
		t.Errorf("Shape = %v, want [4 8 14 14]", y.Shape)
	}
}

func TestMaxPoolHasNoParams(t *testing.T) {
	if ps := NewMaxPool2D(2).Params(); len(ps) != 0 {
		t.Errorf("Params() = %v, want none", ps)
	}
}

func TestMaxPoolIsLayer(t *testing.T) {
	var _ Layer = NewMaxPool2D(2)
}

func TestMaxPoolPanicsOnNon4D(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Forward of a 2d tensor did not panic")
		}
	}()
	NewMaxPool2D(2).Forward(tensor.New(4, 4))
}

// values are a shuffled ladder 0.1 apart, so a 0.01 nudge never changes a window's winner.
func TestMaxPoolGradCheck(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	x := tensor.New(2, 3, 4, 4)
	for i, p := range r.Perm(len(x.Data)) {
		x.Data[i] = float32(p)*0.1 - 4
	}
	gradCheck(t, NewMaxPool2D(2), x, r)
}
