package tensor

import (
	"slices"
	"testing"
)

func TestBroadcastShapes(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want []int
	}{
		{"same shape", []int{2, 3}, []int{2, 3}, []int{2, 3}},
		{"bias onto batch", []int{32, 10}, []int{10}, []int{32, 10}},
		{"bias onto batch, flipped", []int{10}, []int{32, 10}, []int{32, 10}},
		{"column times row", []int{3, 1}, []int{1, 4}, []int{3, 4}},
		{"size-1 stretches", []int{2, 3}, []int{1, 3}, []int{2, 3}},
		{"3d mixed", []int{8, 1, 28}, []int{5, 1}, []int{8, 5, 28}},
		{"scalar-like", []int{2, 3, 4}, []int{1}, []int{2, 3, 4}},
		{"both size 1", []int{1}, []int{1}, []int{1}},
		{"per-channel bias in nchw", []int{4, 16, 7, 7}, []int{16, 1, 1}, []int{4, 16, 7, 7}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := broadcastShapes(tt.a, tt.b); !slices.Equal(got, tt.want) {
				t.Errorf("broadcastShapes(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestBroadcastShapesDoesNotModifyInputs(t *testing.T) {
	a, b := []int{3, 1}, []int{1, 4}
	broadcastShapes(a, b)

	if !slices.Equal(a, []int{3, 1}) || !slices.Equal(b, []int{1, 4}) {
		t.Errorf("inputs changed: a = %v, b = %v", a, b)
	}
}

func TestBroadcastShapesPanics(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
	}{
		{"last dim differs", []int{3, 4}, []int{3, 5}},
		{"first dim differs", []int{3, 4}, []int{2, 4}},
		{"trailing mismatch with shorter shape", []int{32, 10}, []int{9}},
		{"3d middle mismatch", []int{2, 3, 4}, []int{2, 4}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("broadcastShapes(%v, %v) did not panic", tt.a, tt.b)
				}
			}()
			broadcastShapes(tt.a, tt.b)
		})
	}
}
