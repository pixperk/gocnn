package tensor

import "fmt"

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
