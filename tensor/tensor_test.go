package tensor

import (
	"slices"
	"testing"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		wantSize int
	}{
		{"scalar-like", []int{1}, 1},
		{"vector", []int{5}, 5},
		{"matrix", []int{2, 3}, 6},
		{"3d", []int{2, 3, 4}, 24},
		{"batch of images", []int{2, 1, 28, 28}, 1568},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := New(tt.shape...)

			if !slices.Equal(x.Shape, tt.shape) {
				t.Errorf("Shape = %v, want %v", x.Shape, tt.shape)
			}
			if len(x.Data) != tt.wantSize {
				t.Errorf("len(Data) = %d, want %d", len(x.Data), tt.wantSize)
			}
			for i, v := range x.Data {
				if v != 0 {
					t.Fatalf("Data[%d] = %v, want 0", i, v)
				}
			}
		})
	}
}

func TestNewCopiesShape(t *testing.T) {
	shape := []int{2, 3}
	x := New(shape...)

	shape[0] = 99

	if x.Shape[0] != 2 {
		t.Fatalf("tensor shape changed when caller's slice changed: Shape = %v", x.Shape)
	}
}

func TestAt2D(t *testing.T) {
	// [[1 2 3]
	//  [4 5 6]]
	x := New(2, 3)
	copy(x.Data, []float32{1, 2, 3, 4, 5, 6})

	tests := []struct {
		i, j int
		want float32
	}{
		{0, 0, 1},
		{0, 1, 2},
		{0, 2, 3},
		{1, 0, 4},
		{1, 1, 5},
		{1, 2, 6},
	}
	for _, tt := range tests {
		if got := x.At(tt.i, tt.j); got != tt.want {
			t.Errorf("At(%d, %d) = %v, want %v", tt.i, tt.j, got, tt.want)
		}
	}
}

func TestAt3D(t *testing.T) {
	// Shape (2, 3, 4), Data = 0..23, so each element equals its own flat
	// position. The want values are the row-major offsets.
	x := New(2, 3, 4)
	for i := range x.Data {
		x.Data[i] = float32(i)
	}

	tests := []struct {
		idx  []int
		want float32
	}{
		{[]int{0, 0, 0}, 0},
		{[]int{0, 0, 3}, 3},
		{[]int{0, 1, 0}, 4},
		{[]int{0, 2, 3}, 11},
		{[]int{1, 0, 0}, 12},
		{[]int{1, 2, 3}, 23},
		{[]int{1, 1, 2}, 18},
	}
	for _, tt := range tests {
		if got := x.At(tt.idx...); got != tt.want {
			t.Errorf("At(%v) = %v, want %v", tt.idx, got, tt.want)
		}
	}
}

func TestAtPanicsOnBadIndex(t *testing.T) {
	x := New(2, 3)

	tests := []struct {
		name string
		idx  []int
	}{
		{"too few indices", []int{1}},
		{"too many indices", []int{1, 1, 1}},
		{"row out of range", []int{2, 0}},
		{"col out of range", []int{0, 3}},
		{"negative index", []int{-1, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("At(%v) did not panic", tt.idx)
				}
			}()
			x.At(tt.idx...)
		})
	}
}
