package tensor

import (
	"slices"
	"testing"
)

func from(data []float32, shape ...int) *Tensor {
	x := New(shape...)
	copy(x.Data, data)
	return x
}

func TestElementwiseSameShape(t *testing.T) {
	a := from([]float32{1, 2, 3, 4}, 2, 2)
	b := from([]float32{10, 20, 30, 40}, 2, 2)

	tests := []struct {
		name string
		op   func(a, b *Tensor) *Tensor
		want []float32
	}{
		{"Add", Add, []float32{11, 22, 33, 44}},
		{"Sub", Sub, []float32{-9, -18, -27, -36}},
		{"Mul", Mul, []float32{10, 40, 90, 160}},
		{"Div", Div, []float32{0.1, 0.1, 0.1, 0.1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.op(a, b)
			if !slices.Equal(got.Shape, []int{2, 2}) {
				t.Fatalf("Shape = %v, want [2 2]", got.Shape)
			}
			for i := range tt.want {
				if d := got.Data[i] - tt.want[i]; d > 1e-6 || d < -1e-6 {
					t.Errorf("Data = %v, want %v", got.Data, tt.want)
					break
				}
			}
		})
	}
}

func TestAddBroadcast(t *testing.T) {
	tests := []struct {
		name      string
		a, b      *Tensor
		wantShape []int
		want      []float32
	}{
		{
			"bias onto batch",
			from([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
			from([]float32{10, 20, 30}, 3),
			[]int{2, 3},
			[]float32{11, 22, 33, 14, 25, 36},
		},
		{
			"bias first",
			from([]float32{10, 20, 30}, 3),
			from([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
			[]int{2, 3},
			[]float32{11, 22, 33, 14, 25, 36},
		},
		{
			"column times row",
			from([]float32{1, 2, 3}, 3, 1),
			from([]float32{10, 20, 30, 40}, 1, 4),
			[]int{3, 4},
			[]float32{11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43},
		},
		{
			"one value per row",
			from([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
			from([]float32{100, 200}, 2, 1),
			[]int{2, 3},
			[]float32{101, 102, 103, 204, 205, 206},
		},
		{
			"single value everywhere",
			from([]float32{1, 2, 3, 4}, 2, 2),
			from([]float32{0.5}, 1),
			[]int{2, 2},
			[]float32{1.5, 2.5, 3.5, 4.5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Add(tt.a, tt.b)
			if !slices.Equal(got.Shape, tt.wantShape) {
				t.Fatalf("Shape = %v, want %v", got.Shape, tt.wantShape)
			}
			if !slices.Equal(got.Data, tt.want) {
				t.Errorf("Data = %v, want %v", got.Data, tt.want)
			}
		})
	}
}

// a transposed input must be read in logical order, not memory order.
func TestAddNonContiguous(t *testing.T) {
	a := from([]float32{1, 2, 3, 4, 5, 6}, 2, 3).Transpose(0, 1) // [[1 4] [2 5] [3 6]]
	b := from([]float32{10, 20, 30, 40, 50, 60}, 3, 2)

	got := Add(a, b)
	want := []float32{11, 24, 32, 45, 53, 66}
	if !slices.Equal(got.Data, want) {
		t.Errorf("Data = %v, want %v", got.Data, want)
	}
}

func TestElementwiseResultIsFreshAndContiguous(t *testing.T) {
	a := from([]float32{1, 2, 3}, 3)
	b := from([]float32{1, 1, 1, 1, 1, 1}, 2, 3)

	got := Add(a, b)

	if !got.IsContiguous() {
		t.Errorf("result not contiguous: strides %v", got.Strides)
	}
	if !slices.Equal(a.Data, []float32{1, 2, 3}) {
		t.Errorf("input a changed: %v", a.Data)
	}
	if !slices.Equal(b.Data, []float32{1, 1, 1, 1, 1, 1}) {
		t.Errorf("input b changed: %v", b.Data)
	}

	got.Data[0] = 99
	if a.Data[0] == 99 || b.Data[0] == 99 {
		t.Error("result shares memory with an input")
	}
}

func TestElementwisePanicsOnBadShapes(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Add of [3,4] and [2,4] did not panic")
		}
	}()
	Add(New(3, 4), New(2, 4))
}
