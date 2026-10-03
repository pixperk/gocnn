package tensor

import (
	"slices"
	"testing"
)

func TestExpandStrides(t *testing.T) {
	tests := []struct {
		name        string
		from        []int
		to          []int
		wantStrides []int
	}{
		{"same shape keeps strides", []int{2, 3}, []int{2, 3}, []int{3, 1}},
		{"row onto batch", []int{3}, []int{4, 3}, []int{0, 1}},
		{"size-1 row dim", []int{1, 3}, []int{4, 3}, []int{0, 1}},
		{"column sideways", []int{3, 1}, []int{3, 4}, []int{1, 0}},
		{"pad two dims", []int{5}, []int{2, 3, 5}, []int{0, 0, 1}},
		{"per-channel bias", []int{16, 1, 1}, []int{4, 16, 7, 7}, []int{0, 1, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := expand(New(tt.from...), tt.to)

			if !slices.Equal(got.Shape, tt.to) {
				t.Errorf("Shape = %v, want %v", got.Shape, tt.to)
			}
			if !slices.Equal(got.Strides, tt.wantStrides) {
				t.Errorf("Strides = %v, want %v", got.Strides, tt.wantStrides)
			}
		})
	}
}

func TestExpandValues(t *testing.T) {
	col := New(3, 1)
	copy(col.Data, []float32{1, 2, 3})
	x := expand(col, []int{3, 4})

	for i := range 3 {
		for j := range 4 {
			if got, want := x.At(i, j), float32(i+1); got != want {
				t.Errorf("At(%d, %d) = %v, want %v", i, j, got, want)
			}
		}
	}
}

// expanding an already strided view must keep its existing strides
// on the dims it does not stretch.
func TestExpandOfTranspose(t *testing.T) {
	x := New(2, 3)
	copy(x.Data, []float32{1, 2, 3, 4, 5, 6})
	y := expand(x.Transpose(0, 1), []int{2, 3, 2}) // [3,2] padded to [2,3,2]

	if !slices.Equal(y.Strides, []int{0, 1, 3}) {
		t.Fatalf("Strides = %v, want [0 1 3]", y.Strides)
	}
	for b := range 2 {
		if got := y.At(b, 2, 1); got != 6 {
			t.Errorf("At(%d, 2, 1) = %v, want 6", b, got)
		}
	}
}

func TestExpandIsView(t *testing.T) {
	x := New(3)
	y := expand(x, []int{4, 3})

	x.Set(9, 1)
	for i := range 4 {
		if got := y.At(i, 1); got != 9 {
			t.Errorf("y.At(%d, 1) = %v after x.Set(9, 1), want 9 (data not shared)", i, got)
		}
	}
}

func TestExpandDoesNotAliasShape(t *testing.T) {
	shape := []int{4, 3}
	y := expand(New(3), shape)

	shape[0] = 99
	if y.Shape[0] != 4 {
		t.Errorf("Shape = %v after caller changed its slice, want [4 3]", y.Shape)
	}
}

func TestExpandPanics(t *testing.T) {
	tests := []struct {
		name string
		from []int
		to   []int
	}{
		{"size mismatch", []int{3}, []int{4, 5}},
		{"cannot shrink dims", []int{2, 3}, []int{3}},
		{"non-1 cannot stretch", []int{2, 3}, []int{4, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("expand(%v -> %v) did not panic", tt.from, tt.to)
				}
			}()
			expand(New(tt.from...), tt.to)
		})
	}
}
