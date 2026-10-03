package tensor

import (
	"slices"
	"testing"
)

func TestNewStrides(t *testing.T) {
	tests := []struct {
		shape []int
		want  []int
	}{
		{[]int{5}, []int{1}},
		{[]int{2, 3}, []int{3, 1}},
		{[]int{2, 3, 4}, []int{12, 4, 1}},
		{[]int{2, 1, 28, 28}, []int{784, 784, 28, 1}},
		{[]int{3, 1}, []int{1, 1}},
	}
	for _, tt := range tests {
		x := New(tt.shape...)
		if !slices.Equal(x.Strides, tt.want) {
			t.Errorf("New(%v).Strides = %v, want %v", tt.shape, x.Strides, tt.want)
		}
	}
}

// a hand-built transpose: at must follow strides, not recompute them.
func TestAtFollowsStrides(t *testing.T) {
	x := &Tensor{
		Data:    []float32{1, 2, 3, 4, 5, 6},
		Shape:   []int{3, 2},
		Strides: []int{1, 3},
	}

	want := [][]float32{{1, 4}, {2, 5}, {3, 6}}
	for i := range 3 {
		for j := range 2 {
			if got := x.At(i, j); got != want[i][j] {
				t.Errorf("At(%d, %d) = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

// stride 0 repeats one row of 3 values as a 4x3 matrix.
func TestAtZeroStride(t *testing.T) {
	x := &Tensor{
		Data:    []float32{7, 8, 9},
		Shape:   []int{4, 3},
		Strides: []int{0, 1},
	}

	for i := range 4 {
		for j, want := range []float32{7, 8, 9} {
			if got := x.At(i, j); got != want {
				t.Errorf("At(%d, %d) = %v, want %v", i, j, got, want)
			}
		}
	}
}

func TestSet(t *testing.T) {
	x := New(2, 3)
	x.Set(42, 1, 2)

	if got := x.At(1, 2); got != 42 {
		t.Errorf("At(1, 2) after Set = %v, want 42", got)
	}
	if x.Data[5] != 42 {
		t.Errorf("Data[5] = %v, want 42", x.Data[5])
	}
	for i, v := range x.Data[:5] {
		if v != 0 {
			t.Errorf("Data[%d] = %v, want 0 (Set touched the wrong element)", i, v)
		}
	}
}

// set through a strided view writes to the underlying data.
func TestSetFollowsStrides(t *testing.T) {
	x := &Tensor{
		Data:    []float32{1, 2, 3, 4, 5, 6},
		Shape:   []int{3, 2},
		Strides: []int{1, 3},
	}
	x.Set(99, 2, 0) // logical (2,0) is data[2*1 + 0*3] = data[2]

	want := []float32{1, 2, 99, 4, 5, 6}
	if !slices.Equal(x.Data, want) {
		t.Errorf("Data = %v, want %v", x.Data, want)
	}
}

func TestSetPanicsOnBadIndex(t *testing.T) {
	x := New(2, 3)

	tests := []struct {
		name string
		idx  []int
	}{
		{"too few indices", []int{1}},
		{"too many indices", []int{1, 1, 1}},
		{"out of range", []int{0, 3}},
		{"negative index", []int{-1, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("Set(%v) did not panic", tt.idx)
				}
			}()
			x.Set(1, tt.idx...)
		})
	}
}
