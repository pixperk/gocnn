package tensor

import (
	"slices"
	"testing"
)

// fill sets data to 0, 1, 2, ... so each value is its row-major position.
func fill(x *Tensor) *Tensor {
	for i := range x.Data {
		x.Data[i] = float32(i)
	}
	return x
}

func TestTranspose2D(t *testing.T) {
	x := New(2, 3)
	copy(x.Data, []float32{1, 2, 3, 4, 5, 6})
	y := x.Transpose(0, 1)

	if !slices.Equal(y.Shape, []int{3, 2}) {
		t.Errorf("Shape = %v, want [3 2]", y.Shape)
	}
	if !slices.Equal(y.Strides, []int{1, 3}) {
		t.Errorf("Strides = %v, want [1 3]", y.Strides)
	}

	want := [][]float32{{1, 4}, {2, 5}, {3, 6}}
	for i := range 3 {
		for j := range 2 {
			if got := y.At(i, j); got != want[i][j] {
				t.Errorf("At(%d, %d) = %v, want %v", i, j, got, want[i][j])
			}
		}
	}
}

func TestTranspose3D(t *testing.T) {
	x := fill(New(2, 3, 4))
	y := x.Transpose(0, 2)

	if !slices.Equal(y.Shape, []int{4, 3, 2}) {
		t.Fatalf("Shape = %v, want [4 3 2]", y.Shape)
	}
	for i := range 2 {
		for j := range 3 {
			for k := range 4 {
				if got, want := y.At(k, j, i), x.At(i, j, k); got != want {
					t.Errorf("y.At(%d,%d,%d) = %v, want x.At(%d,%d,%d) = %v", k, j, i, got, i, j, k, want)
				}
			}
		}
	}
}

func TestTransposeIsView(t *testing.T) {
	x := New(2, 3)
	y := x.Transpose(0, 1)

	if !slices.Equal(x.Shape, []int{2, 3}) || !slices.Equal(x.Strides, []int{3, 1}) {
		t.Errorf("original changed: Shape %v Strides %v", x.Shape, x.Strides)
	}

	y.Set(42, 2, 1) // y(2,1) is x(1,2)
	if got := x.At(1, 2); got != 42 {
		t.Errorf("x.At(1, 2) = %v after y.Set(42, 2, 1), want 42 (data not shared)", got)
	}
}

func TestTransposeTwiceIsIdentity(t *testing.T) {
	x := fill(New(2, 3, 4))
	y := x.Transpose(0, 2).Transpose(0, 2)

	if !slices.Equal(y.Shape, x.Shape) || !slices.Equal(y.Strides, x.Strides) {
		t.Errorf("got Shape %v Strides %v, want %v %v", y.Shape, y.Strides, x.Shape, x.Strides)
	}
}

func TestTransposePanics(t *testing.T) {
	tests := []struct {
		name string
		a, b int
	}{
		{"a too large", 2, 0},
		{"b too large", 0, 2},
		{"negative", -1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("Transpose(%d, %d) on shape [2 3] did not panic", tt.a, tt.b)
				}
			}()
			New(2, 3).Transpose(tt.a, tt.b)
		})
	}
}

func TestIsContiguous(t *testing.T) {
	tests := []struct {
		name string
		x    *Tensor
		want bool
	}{
		{"fresh", New(2, 3), true},
		{"reshaped", New(2, 3).Reshape(3, 2), true},
		{"transposed", New(2, 3).Transpose(0, 1), false},
		{"transposed back", New(2, 3).Transpose(0, 1).Transpose(0, 1), true},
		{"broadcast", &Tensor{Data: []float32{1, 2, 3}, Shape: []int{4, 3}, Strides: []int{0, 1}}, false},
	}
	for _, tt := range tests {
		if got := tt.x.IsContiguous(); got != tt.want {
			t.Errorf("%s: IsContiguous() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestContiguousReturnsSameWhenAlreadyContiguous(t *testing.T) {
	x := New(2, 3)
	if x.Contiguous() != x {
		t.Error("Contiguous() copied a tensor that was already contiguous")
	}
}

func TestContiguousOfTranspose(t *testing.T) {
	x := New(2, 3)
	copy(x.Data, []float32{1, 2, 3, 4, 5, 6})
	y := x.Transpose(0, 1).Contiguous()

	if !y.IsContiguous() {
		t.Fatalf("result not contiguous: Shape %v Strides %v", y.Shape, y.Strides)
	}
	if want := []float32{1, 4, 2, 5, 3, 6}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}

	y.Set(99, 0, 0)
	if x.At(0, 0) != 1 {
		t.Errorf("x.At(0, 0) = %v after writing to the copy, want 1", x.At(0, 0))
	}
}

func TestContiguous3D(t *testing.T) {
	x := fill(New(2, 3, 4))
	y := x.Transpose(1, 2).Contiguous() // shape [2 4 3]

	if !slices.Equal(y.Shape, []int{2, 4, 3}) {
		t.Fatalf("Shape = %v, want [2 4 3]", y.Shape)
	}
	for i := range 2 {
		for j := range 4 {
			for k := range 3 {
				if got, want := y.At(i, j, k), x.At(i, k, j); got != want {
					t.Errorf("y.At(%d,%d,%d) = %v, want %v", i, j, k, got, want)
				}
			}
		}
	}
}

func TestContiguousOfBroadcast(t *testing.T) {
	x := &Tensor{Data: []float32{7, 8, 9}, Shape: []int{4, 3}, Strides: []int{0, 1}}
	y := x.Contiguous()

	want := []float32{7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9}
	if !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

// reshape refuses a transpose but accepts its contiguous copy.
func TestReshapeAfterContiguous(t *testing.T) {
	x := New(2, 3)
	copy(x.Data, []float32{1, 2, 3, 4, 5, 6})
	y := x.Transpose(0, 1).Contiguous().Reshape(-1)

	if want := []float32{1, 4, 2, 5, 3, 6}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}
