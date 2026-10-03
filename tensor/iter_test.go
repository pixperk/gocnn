package tensor

import (
	"slices"
	"testing"
)

// walks a normal and a broadcast tensor together.
func TestOdometerTwoTensors(t *testing.T) {
	a := New(3, 2)
	b := expand(New(2), []int{3, 2}) // strides [0 1]

	o := newOdometer([]int{3, 2}, a, b)

	wantA := []int{0, 1, 2, 3, 4, 5}
	wantB := []int{0, 1, 0, 1, 0, 1}
	for i := range 6 {
		if o.offs[0] != wantA[i] || o.offs[1] != wantB[i] {
			t.Errorf("step %d: offs = %v, want [%d %d]", i, o.offs, wantA[i], wantB[i])
		}
		o.next()
	}
}

func TestOdometerTransposed(t *testing.T) {
	x := New(2, 3).Transpose(0, 1) // shape [3 2], strides [1 3]
	o := newOdometer(x.Shape, x)

	want := []int{0, 3, 1, 4, 2, 5}
	var got []int
	for range 6 {
		got = append(got, o.offs[0])
		o.next()
	}
	if !slices.Equal(got, want) {
		t.Errorf("offsets = %v, want %v", got, want)
	}
}

func TestOdometer3DCarry(t *testing.T) {
	x := New(2, 2, 3)
	o := newOdometer(x.Shape, x)

	for i := range 12 {
		if o.offs[0] != i {
			t.Fatalf("step %d: offset %d, want %d", i, o.offs[0], i)
		}
		o.next()
	}
}

func TestOdometerPanicsOnShapeMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("newOdometer with mismatched shapes did not panic")
		}
	}()
	newOdometer([]int{3, 2}, New(3, 2), New(2))
}
