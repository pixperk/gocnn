package tensor

import (
	"fmt"
	"slices"
)

// odometer steps through every index of a shape in row-major order,
// tracking each tensor's data offset for the current index.
type odometer struct {
	shape   []int
	idx     []int
	strides [][]int // strides[k] belongs to tensor k
	offs    []int   // offs[k] is tensor k's data offset at idx
}

// newOdometer starts at index (0, 0, ...). every tensor must already
// have the given shape; expand them first if they don't.
func newOdometer(shape []int, ts ...*Tensor) *odometer {
	o := &odometer{
		shape:   shape,
		idx:     make([]int, len(shape)),
		strides: make([][]int, len(ts)),
		offs:    make([]int, len(ts)),
	}
	for k, t := range ts {
		if !slices.Equal(t.Shape, shape) {
			panic(fmt.Sprintf("tensor: odometer over %v got tensor of shape %v", shape, t.Shape))
		}
		o.strides[k] = t.Strides
	}
	return o
}

// next bumps the last dim and carries left when it wraps.
func (o *odometer) next() {
	for d := len(o.idx) - 1; d >= 0; d-- {
		o.idx[d]++
		for k, s := range o.strides {
			o.offs[k] += s[d]
		}
		if o.idx[d] < o.shape[d] {
			return
		}
		for k, s := range o.strides {
			o.offs[k] -= o.idx[d] * s[d]
		}
		o.idx[d] = 0
	}
}
