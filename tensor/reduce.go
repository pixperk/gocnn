package tensor

import (
	"fmt"
	"math"
	"slices"
)

var negInf = float32(math.Inf(-1))

// sum adds up values along axis.
func (t *Tensor) Sum(axis int, keepDims bool) *Tensor {
	return reduce(t, axis, keepDims, 0, func(acc, x float32) float32 { return acc + x })
}

// max takes the largest value along axis.
func (t *Tensor) Max(axis int, keepDims bool) *Tensor {
	return reduce(t, axis, keepDims, negInf, func(acc, x float32) float32 { return max(acc, x) })
}

// argmax returns the position of the largest value along axis; ties go to the first.
func (t *Tensor) ArgMax(axis int) []int {
	best, eb := reduceTarget(t, axis, negInf)
	pos := make([]int, len(best.Data))

	o := newOdometer(t.Shape, t, eb)
	for range numel(t.Shape) {
		slot := o.offs[1]
		if v := t.Data[o.offs[0]]; v > best.Data[slot] {
			best.Data[slot] = v
			pos[slot] = o.idx[axis]
		}
		o.next()
	}
	return pos
}

// reduce folds every value along axis into one slot with f, starting at init.
func reduce(t *Tensor, axis int, keepDims bool, init float32, f func(acc, x float32) float32) *Tensor {
	out, eo := reduceTarget(t, axis, init)

	o := newOdometer(t.Shape, t, eo)
	for range numel(t.Shape) {
		eo.Data[o.offs[1]] = f(eo.Data[o.offs[1]], t.Data[o.offs[0]])
		o.next()
	}

	if keepDims {
		return out
	}
	return out.Reshape(slices.Delete(slices.Clone(out.Shape), axis, axis+1)...)
}

// reduceTarget makes the reduction output and its view expanded over axis.
func reduceTarget(t *Tensor, axis int, init float32) (out, expanded *Tensor) {
	if axis < 0 || axis >= len(t.Shape) {
		panic(fmt.Sprintf("tensor: axis %d out of range for shape %v", axis, t.Shape))
	}

	shape := slices.Clone(t.Shape)
	shape[axis] = 1
	out = New(shape...)
	for i := range out.Data {
		out.Data[i] = init
	}
	return out, expand(out, t.Shape)
}
