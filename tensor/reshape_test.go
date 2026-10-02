package tensor

import (
	"slices"
	"testing"
)

func TestReshape(t *testing.T) {
	tests := []struct {
		name        string
		from        []int
		to          []int
		wantShape   []int
		wantStrides []int
	}{
		{"matrix to matrix", []int{2, 3}, []int{3, 2}, []int{3, 2}, []int{2, 1}},
		{"matrix to vector", []int{2, 3}, []int{6}, []int{6}, []int{1}},
		{"vector to matrix", []int{6}, []int{2, 3}, []int{2, 3}, []int{3, 1}},
		{"add a size-1 dim", []int{2, 3}, []int{1, 2, 3}, []int{1, 2, 3}, []int{6, 3, 1}},
		{"infer last", []int{2, 3}, []int{3, -1}, []int{3, 2}, []int{2, 1}},
		{"infer first", []int{2, 3}, []int{-1, 2}, []int{3, 2}, []int{2, 1}},
		{"infer only dim", []int{2, 3}, []int{-1}, []int{6}, []int{1}},
		{"flatten batch", []int{4, 16, 7, 7}, []int{4, -1}, []int{4, 784}, []int{784, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := New(tt.from...)
			y := x.Reshape(tt.to...)

			if !slices.Equal(y.Shape, tt.wantShape) {
				t.Errorf("Shape = %v, want %v", y.Shape, tt.wantShape)
			}
			if !slices.Equal(y.Strides, tt.wantStrides) {
				t.Errorf("Strides = %v, want %v", y.Strides, tt.wantStrides)
			}
		})
	}
}

func TestReshapeKeepsOrder(t *testing.T) {
	x := New(2, 3)
	copy(x.Data, []float32{1, 2, 3, 4, 5, 6})
	y := x.Reshape(3, 2)

	// [[1 2]
	//  [3 4]
	//  [5 6]]
	want := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	for i := range 3 {
		for j := range 2 {
			if got := y.At(i, j); got != want[i][j] {
				t.Errorf("At(%d, %d) = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

// reshape returns a new view; the original keeps its shape.
func TestReshapeDoesNotModifyOriginal(t *testing.T) {
	x := New(2, 3)
	y := x.Reshape(3, 2)

	if x == y {
		t.Fatal("Reshape returned the same *Tensor; want a new view")
	}
	if !slices.Equal(x.Shape, []int{2, 3}) {
		t.Errorf("original Shape = %v, want [2 3]", x.Shape)
	}
	if !slices.Equal(x.Strides, []int{3, 1}) {
		t.Errorf("original Strides = %v, want [3 1]", x.Strides)
	}
}

// the view shares memory: a write through one is visible in the other.
func TestReshapeSharesData(t *testing.T) {
	x := New(2, 3)
	y := x.Reshape(6)

	y.Set(42, 4)

	if got := x.At(1, 1); got != 42 {
		t.Errorf("x.At(1, 1) = %v after y.Set(42, 4), want 42 (data not shared)", got)
	}
}

func TestReshapeDoesNotModifyCallerShape(t *testing.T) {
	x := New(2, 3)
	shape := []int{-1, 2}

	y := x.Reshape(shape...)

	if !slices.Equal(shape, []int{-1, 2}) {
		t.Errorf("caller's slice = %v, want [-1 2] (reshape wrote into it)", shape)
	}

	shape[1] = 99
	if !slices.Equal(y.Shape, []int{3, 2}) {
		t.Errorf("y.Shape = %v after caller changed its slice, want [3 2]", y.Shape)
	}
}

func TestReshapeChained(t *testing.T) {
	x := New(2, 3, 4)
	for i := range x.Data {
		x.Data[i] = float32(i)
	}
	y := x.Reshape(6, 4).Reshape(-1).Reshape(4, 3, 2)

	if got := y.At(3, 2, 1); got != 23 {
		t.Errorf("At(3, 2, 1) = %v, want 23", got)
	}
	if got := y.At(1, 0, 1); got != 7 {
		t.Errorf("At(1, 0, 1) = %v, want 7", got)
	}
}

func TestReshapePanics(t *testing.T) {
	tests := []struct {
		name string
		to   []int
	}{
		{"too many elements", []int{4, 2}},
		{"too few elements", []int{2, 2}},
		{"two inferred dims", []int{-1, -1}},
		{"inferred does not divide", []int{-1, 4}},
		{"zero dim", []int{0, 6}},
		{"negative dims with right product", []int{-2, -3}},
		{"negative dim other than -1", []int{-2, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := New(2, 3)
			defer func() {
				if recover() == nil {
					t.Errorf("Reshape(%v) of shape [2 3] did not panic", tt.to)
				}
			}()
			x.Reshape(tt.to...)
		})
	}
}

// reshape only reinterprets memory, so it must refuse tensors whose
// logical order differs from their memory order.
func TestReshapeNonContiguousPanics(t *testing.T) {
	tests := []struct {
		name string
		x    *Tensor
	}{
		{"transposed", &Tensor{
			Data:    []float32{1, 2, 3, 4, 5, 6},
			Shape:   []int{3, 2},
			Strides: []int{1, 3},
		}},
		{"broadcast", &Tensor{
			Data:    []float32{7, 8, 9},
			Shape:   []int{4, 3},
			Strides: []int{0, 1},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("Reshape of %s tensor did not panic", tt.name)
				}
			}()
			tt.x.Reshape(-1)
		})
	}
}
