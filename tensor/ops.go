package tensor

import (
	"fmt"
	"slices"
)

// broadcastShapes returns the shape of a op b under numpy rules:
// align from the right, each dim pair must be equal or contain a 1.
func broadcastShapes(a, b []int) []int {
	n := max(len(a), len(b))
	out := make([]int, n)

	for i := range n { // i counts from the right
		da := dimFromRight(a, i)
		db := dimFromRight(b, i)

		switch {
		case da == db, db == 1:
			out[n-1-i] = da
		case da == 1:
			out[n-1-i] = db
		default:
			panic(fmt.Sprintf("tensor: cannot broadcast shapes %v and %v", a, b))
		}
	}
	return out
}

// dimFromRight returns shape's i-th dim counting from the right,
// or 1 if shape is too short.
func dimFromRight(shape []int, i int) int {
	if i >= len(shape) {
		return 1
	}
	return shape[len(shape)-1-i]
}

// expand returns a view of t stretched to shape. no copy: stretched
// and padded dims get stride 0, so they reread the same memory.
func expand(t *Tensor, shape []int) *Tensor {
	if len(t.Shape) > len(shape) {
		panic(fmt.Sprintf("tensor: cannot expand %v to fewer dims %v", t.Shape, shape))
	}

	strides := make([]int, len(shape))
	for i := range shape {
		out := len(shape) - 1 - i   // dim in the result
		src := len(t.Shape) - 1 - i // matching dim in t, < 0 means padding

		switch {
		case src < 0:
			strides[out] = 0
		case t.Shape[src] == shape[out]:
			strides[out] = t.Strides[src]
		case t.Shape[src] == 1:
			strides[out] = 0
		default:
			panic(fmt.Sprintf("tensor: cannot expand %v to %v", t.Shape, shape))
		}
	}

	return &Tensor{Data: t.Data, Shape: slices.Clone(shape), Strides: strides}
}
