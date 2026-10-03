package tensor

import (
	"slices"
	"testing"
)

func TestSum(t *testing.T) {
	x := from([]float32{1, 2, 3, 4, 5, 6}, 2, 3)

	tests := []struct {
		name      string
		axis      int
		keepDims  bool
		wantShape []int
		want      []float32
	}{
		{"rows", 1, false, []int{2}, []float32{6, 15}},
		{"rows keepdims", 1, true, []int{2, 1}, []float32{6, 15}},
		{"cols", 0, false, []int{3}, []float32{5, 7, 9}},
		{"cols keepdims", 0, true, []int{1, 3}, []float32{5, 7, 9}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := x.Sum(tt.axis, tt.keepDims)
			if !slices.Equal(got.Shape, tt.wantShape) {
				t.Fatalf("Shape = %v, want %v", got.Shape, tt.wantShape)
			}
			if !slices.Equal(got.Data, tt.want) {
				t.Errorf("Data = %v, want %v", got.Data, tt.want)
			}
		})
	}
}

func TestSum3DMiddleAxis(t *testing.T) {
	x := fill(New(2, 3, 4)) // 0..23
	got := x.Sum(1, false)

	if !slices.Equal(got.Shape, []int{2, 4}) {
		t.Fatalf("Shape = %v, want [2 4]", got.Shape)
	}
	// block 0, col 0: 0+4+8 = 12; block 1, col 3: 15+19+23 = 57
	want := []float32{12, 15, 18, 21, 48, 51, 54, 57}
	if !slices.Equal(got.Data, want) {
		t.Errorf("Data = %v, want %v", got.Data, want)
	}
}

func TestSumOf1DGivesScalarShape(t *testing.T) {
	got := from([]float32{1, 2, 3}, 3).Sum(0, false)

	if len(got.Shape) != 0 {
		t.Fatalf("Shape = %v, want [] (0-dim)", got.Shape)
	}
	if got.At() != 6 {
		t.Errorf("At() = %v, want 6", got.At())
	}
}

func TestSumNonContiguous(t *testing.T) {
	x := from([]float32{1, 2, 3, 4, 5, 6}, 2, 3).Transpose(0, 1) // [[1 4] [2 5] [3 6]]
	got := x.Sum(1, false)

	if want := []float32{5, 7, 9}; !slices.Equal(got.Data, want) {
		t.Errorf("Data = %v, want %v", got.Data, want)
	}
}

func TestSumDoesNotModifyInput(t *testing.T) {
	x := from([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	x.Sum(1, true)

	if want := []float32{1, 2, 3, 4, 5, 6}; !slices.Equal(x.Data, want) {
		t.Errorf("input changed: %v", x.Data)
	}
}

func TestMax(t *testing.T) {
	x := from([]float32{1, 9, 3, 7, 5, 6}, 2, 3)

	got := x.Max(1, false)
	if want := []float32{9, 7}; !slices.Equal(got.Data, want) {
		t.Errorf("Max(1) = %v, want %v", got.Data, want)
	}

	got = x.Max(0, false)
	if want := []float32{7, 9, 6}; !slices.Equal(got.Data, want) {
		t.Errorf("Max(0) = %v, want %v", got.Data, want)
	}
}

// a start value of 0 would wrongly report 0 here.
func TestMaxAllNegative(t *testing.T) {
	x := from([]float32{-5, -2, -9, -1, -8, -3}, 2, 3)
	got := x.Max(1, false)

	if want := []float32{-2, -1}; !slices.Equal(got.Data, want) {
		t.Errorf("Data = %v, want %v", got.Data, want)
	}
}

// the softmax pattern: subtract each row's max from that row.
func TestMaxKeepDimsBroadcastsBack(t *testing.T) {
	x := from([]float32{1, 9, 3, 7, 5, 6}, 2, 3)
	got := Sub(x, x.Max(1, true))

	want := []float32{-8, 0, -6, 0, -2, -1}
	if !slices.Equal(got.Data, want) {
		t.Errorf("Data = %v, want %v", got.Data, want)
	}
}

func TestArgMax(t *testing.T) {
	x := from([]float32{
		0.1, 0.7, 0.2,
		0.5, 0.3, 0.2,
		0.1, 0.1, 0.8,
		-3, -1, -2,
	}, 4, 3)

	if got, want := x.ArgMax(1), []int{1, 0, 2, 1}; !slices.Equal(got, want) {
		t.Errorf("ArgMax(1) = %v, want %v", got, want)
	}
	if got, want := x.ArgMax(0), []int{1, 0, 2}; !slices.Equal(got, want) {
		t.Errorf("ArgMax(0) = %v, want %v", got, want)
	}
}

// on a tie, the first index wins (same as numpy and pytorch).
func TestArgMaxTieTakesFirst(t *testing.T) {
	x := from([]float32{3, 5, 5, 1}, 1, 4)

	if got := x.ArgMax(1); !slices.Equal(got, []int{1}) {
		t.Errorf("ArgMax = %v, want [1]", got)
	}
}

func TestReductionsPanicOnBadAxis(t *testing.T) {
	x := New(2, 3)
	tests := []struct {
		name string
		call func()
	}{
		{"Sum axis 2", func() { x.Sum(2, false) }},
		{"Sum axis -1", func() { x.Sum(-1, false) }},
		{"Max axis 2", func() { x.Max(2, false) }},
		{"ArgMax axis 2", func() { x.ArgMax(2) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", tt.name)
				}
			}()
			tt.call()
		})
	}
}
