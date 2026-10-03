package nn

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

func TestFlattenForward(t *testing.T) {
	x := tensor.New(2, 3, 2, 2)
	for i := range x.Data {
		x.Data[i] = float32(i)
	}

	y := (&Flatten{}).Forward(x)

	if !slices.Equal(y.Shape, []int{2, 12}) {
		t.Fatalf("Shape = %v, want [2 12]", y.Shape)
	}
	if !slices.Equal(y.Data, x.Data) {
		t.Errorf("values or order changed: %v", y.Data)
	}
}

func TestFlattenAlreadyFlat(t *testing.T) {
	y := (&Flatten{}).Forward(tensor.New(5, 7))

	if !slices.Equal(y.Shape, []int{5, 7}) {
		t.Errorf("Shape = %v, want [5 7]", y.Shape)
	}
}

func TestFlattenBackward(t *testing.T) {
	f := &Flatten{}
	f.Forward(tensor.New(2, 3, 2, 2))

	dy := tensor.New(2, 12)
	for i := range dy.Data {
		dy.Data[i] = float32(i) * 0.5
	}
	dx := f.Backward(dy)

	if !slices.Equal(dx.Shape, []int{2, 3, 2, 2}) {
		t.Fatalf("dx shape = %v, want [2 3 2 2]", dx.Shape)
	}
	if !slices.Equal(dx.Data, dy.Data) {
		t.Errorf("dx values changed: %v", dx.Data)
	}
}

func TestFlattenNonContiguous(t *testing.T) {
	x := from([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3).Transpose(1, 2) // [1, 3, 2]: [[1 4] [2 5] [3 6]]

	y := (&Flatten{}).Forward(x)

	if want := []float32{1, 4, 2, 5, 3, 6}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

// changing the input's shape slice later must not change what backward restores.
func TestFlattenKeepsOwnShape(t *testing.T) {
	f := &Flatten{}
	x := tensor.New(2, 3, 4)
	f.Forward(x)
	x.Shape[1] = 99

	if dx := f.Backward(tensor.New(2, 12)); !slices.Equal(dx.Shape, []int{2, 3, 4}) {
		t.Errorf("dx shape = %v, want [2 3 4]", dx.Shape)
	}
}

func TestFlattenHasNoParams(t *testing.T) {
	if ps := (&Flatten{}).Params(); len(ps) != 0 {
		t.Errorf("Params() = %v, want none", ps)
	}
}

func TestFlattenIsLayer(t *testing.T) {
	var _ Layer = &Flatten{}
}

func TestFlattenGradCheck(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	x := tensor.New(2, 3, 2, 2)
	for i := range x.Data {
		x.Data[i] = r.Float32()*2 - 1
	}
	gradCheck(t, &Flatten{}, x, r)
}
