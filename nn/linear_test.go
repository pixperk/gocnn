package nn

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

func from(data []float32, shape ...int) *tensor.Tensor {
	x := tensor.New(shape...)
	copy(x.Data, data)
	return x
}

func TestNewLinearShapes(t *testing.T) {
	l := NewLinear(784, 10, rand.New(rand.NewPCG(1, 2)))

	if !slices.Equal(l.W.Value.Shape, []int{784, 10}) {
		t.Errorf("W shape = %v, want [784 10]", l.W.Value.Shape)
	}
	if !slices.Equal(l.B.Value.Shape, []int{10}) {
		t.Errorf("B shape = %v, want [10]", l.B.Value.Shape)
	}
	if !slices.Equal(l.W.Grad.Shape, l.W.Value.Shape) || !slices.Equal(l.B.Grad.Shape, l.B.Value.Shape) {
		t.Errorf("grad shapes %v %v must match value shapes", l.W.Grad.Shape, l.B.Grad.Shape)
	}
}

// he init: weights ~ normal with std sqrt(2/in), bias all zero.
func TestNewLinearInit(t *testing.T) {
	l := NewLinear(784, 100, rand.New(rand.NewPCG(1, 2)))

	var sum, sumSq float64
	for _, v := range l.W.Value.Data {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}
	n := float64(len(l.W.Value.Data))
	mean := sum / n
	std := math.Sqrt(sumSq/n - mean*mean)
	wantStd := math.Sqrt(2.0 / 784)

	if math.Abs(mean) > 0.005 {
		t.Errorf("W mean = %.4f, want ~0", mean)
	}
	if math.Abs(std-wantStd)/wantStd > 0.05 {
		t.Errorf("W std = %.4f, want ~%.4f", std, wantStd)
	}
	for i, v := range l.B.Value.Data {
		if v != 0 {
			t.Fatalf("B[%d] = %v, want 0", i, v)
		}
	}
}

func TestNewLinearIsSeeded(t *testing.T) {
	a := NewLinear(5, 3, rand.New(rand.NewPCG(7, 7)))
	b := NewLinear(5, 3, rand.New(rand.NewPCG(7, 7)))

	if !slices.Equal(a.W.Value.Data, b.W.Value.Data) {
		t.Error("same seed gave different weights")
	}
}

func TestLinearForward(t *testing.T) {
	l := NewLinear(3, 2, rand.New(rand.NewPCG(1, 2)))
	copy(l.W.Value.Data, []float32{
		1, 0,
		0, 1,
		1, 1,
	})
	copy(l.B.Value.Data, []float32{10, 20})

	x := from([]float32{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)

	y := l.Forward(x)

	// row 0: [1+3, 2+3] + [10 20] = [14 25]
	want := []float32{14, 25, 20, 31}
	if !slices.Equal(y.Shape, []int{2, 2}) {
		t.Fatalf("Shape = %v, want [2 2]", y.Shape)
	}
	if !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

// backward needs x, so forward must keep it.
func TestLinearForwardCachesInput(t *testing.T) {
	l := NewLinear(3, 2, rand.New(rand.NewPCG(1, 2)))
	x := tensor.New(4, 3)

	l.Forward(x)

	if l.x != x {
		t.Error("Forward did not cache its input in l.x")
	}
}

func TestLinearParams(t *testing.T) {
	l := NewLinear(3, 2, rand.New(rand.NewPCG(1, 2)))
	ps := l.Params()

	if len(ps) != 2 || ps[0] != l.W || ps[1] != l.B {
		t.Errorf("Params() = %v, want [W B]", ps)
	}
}

func TestLinearIsLayer(t *testing.T) {
	var _ Layer = NewLinear(1, 1, rand.New(rand.NewPCG(1, 2)))
}

func TestLinearBackwardByHand(t *testing.T) {
	l := NewLinear(2, 1, rand.New(rand.NewPCG(1, 2)))
	copy(l.W.Value.Data, []float32{3, 4})
	copy(l.B.Value.Data, []float32{5})

	l.Forward(from([]float32{1, 2}, 1, 2))
	dx := l.Backward(from([]float32{2}, 1, 1))

	if want := []float32{2, 4}; !slices.Equal(l.W.Grad.Data, want) {
		t.Errorf("dW = %v, want %v", l.W.Grad.Data, want)
	}
	if want := []float32{2}; !slices.Equal(l.B.Grad.Data, want) {
		t.Errorf("db = %v, want %v", l.B.Grad.Data, want)
	}
	if want := []float32{6, 8}; !slices.Equal(dx.Data, want) {
		t.Errorf("dx = %v, want %v", dx.Data, want)
	}
}

func TestLinearBackwardShapes(t *testing.T) {
	l := NewLinear(784, 10, rand.New(rand.NewPCG(1, 2)))
	l.Forward(tensor.New(32, 784))
	dx := l.Backward(tensor.New(32, 10))

	if !slices.Equal(dx.Shape, []int{32, 784}) {
		t.Errorf("dx shape = %v, want [32 784]", dx.Shape)
	}
	if !slices.Equal(l.W.Grad.Shape, []int{784, 10}) {
		t.Errorf("dW shape = %v, want [784 10]", l.W.Grad.Shape)
	}
	if !slices.Equal(l.B.Grad.Shape, []int{10}) {
		t.Errorf("db shape = %v, want [10]", l.B.Grad.Shape)
	}
}

// the batch's blame on a weight adds up across images.
func TestLinearBackwardSumsOverBatch(t *testing.T) {
	l := NewLinear(1, 1, rand.New(rand.NewPCG(1, 2)))
	l.Forward(from([]float32{1, 2, 3}, 3, 1))
	l.Backward(from([]float32{1, 1, 1}, 3, 1))

	if got := l.W.Grad.Data[0]; got != 6 {
		t.Errorf("dW = %v, want 1+2+3 = 6", got)
	}
	if got := l.B.Grad.Data[0]; got != 3 {
		t.Errorf("db = %v, want 1+1+1 = 3", got)
	}
}

func TestLinearBackwardAccumulates(t *testing.T) {
	l := NewLinear(2, 1, rand.New(rand.NewPCG(1, 2)))
	copy(l.W.Value.Data, []float32{3, 4})
	x, dy := from([]float32{1, 2}, 1, 2), from([]float32{2}, 1, 1)

	l.Forward(x)
	l.Backward(dy)
	l.Backward(dy)

	if want := []float32{4, 8}; !slices.Equal(l.W.Grad.Data, want) {
		t.Errorf("dW after two backwards = %v, want %v (grads must add up)", l.W.Grad.Data, want)
	}
	if want := []float32{4}; !slices.Equal(l.B.Grad.Data, want) {
		t.Errorf("db after two backwards = %v, want %v", l.B.Grad.Data, want)
	}
}

func TestLinearGradCheck(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	l := NewLinear(6, 4, r)
	x := tensor.New(5, 6)
	for i := range x.Data {
		x.Data[i] = r.Float32()*2 - 1
	}
	gradCheck(t, l, x, r)
}
