package tensor

import (
	"fmt"
	"slices"
)

type Tensor struct {
	//data is just a slice represention tensor's data
	Data []float32
	//example : shape = [2,3] means 2 rows and 3 columns
	Shape []int
	//number of elements to skip in a dim to get to the next element in that dim.
	Strides []int
}

// new tensor with some shape
func New(shape ...int) *Tensor {
	return &Tensor{
		Data:    make([]float32, numel(shape)),
		Shape:   append([]int(nil), shape...),
		Strides: rowMajorStrides(shape),
	}
}

// rowMajorStrides walks the shape right to left: last stride is 1,
// each earlier one is the next stride times the next dim's size.
func rowMajorStrides(shape []int) []int {
	strides := make([]int, len(shape))
	step := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = step
		step *= shape[i]
	}
	return strides
}

// at returns the element at idx, one index per dim.
func (t *Tensor) At(idx ...int) float32 {
	return t.Data[t.offset(idx)]
}

// set writes v at idx, one index per dim.
func (t *Tensor) Set(v float32, idx ...int) {
	t.Data[t.offset(idx)] = v
}

// reshape returns a view of the same data with a new shape. no copy.
// one dim may be -1 and is inferred. panics if t is not contiguous.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	if !t.IsContiguous() {
		panic(fmt.Sprintf("tensor: reshape of non-contiguous tensor, shape %v strides %v",
			t.Shape, t.Strides))
	}

	newShape := append([]int(nil), shape...)
	inferAt := -1 // index of the -1 dim, if any
	known := 1    // product of all other dims
	for i, s := range newShape {
		switch {
		case s == -1 && inferAt != -1:
			panic(fmt.Sprintf("tensor: reshape to %v has more than one -1", shape))
		case s == -1:
			inferAt = i
		case s < 1:
			panic(fmt.Sprintf("tensor: reshape to %v has invalid dim %d", shape, s))
		default:
			known *= s
		}
	}

	total := numel(t.Shape)
	if inferAt != -1 {
		if total%known != 0 {
			panic(fmt.Sprintf("tensor: cannot reshape %v to %v", t.Shape, shape))
		}
		newShape[inferAt] = total / known
	}

	if numel(newShape) != total {
		panic(fmt.Sprintf("tensor: cannot reshape %v (%d elements) to %v (%d elements)",
			t.Shape, total, newShape, numel(newShape)))
	}

	return &Tensor{
		Data:    t.Data,
		Shape:   newShape,
		Strides: rowMajorStrides(newShape),
	}
}

// transpose returns a view with dims a and b swapped. no copy.
func (t *Tensor) Transpose(a, b int) *Tensor {
	n := len(t.Shape)
	if a < 0 || a >= n || b < 0 || b >= n {
		panic(fmt.Sprintf("tensor: transpose dims %d, %d out of range for shape %v", a, b, t.Shape))
	}

	shape := slices.Clone(t.Shape)
	strides := slices.Clone(t.Strides)
	shape[a], shape[b] = shape[b], shape[a]
	strides[a], strides[b] = strides[b], strides[a]

	return &Tensor{Data: t.Data, Shape: shape, Strides: strides}
}

// iscontiguous reports whether data is laid out in row-major order.
func (t *Tensor) IsContiguous() bool {
	return slices.Equal(t.Strides, rowMajorStrides(t.Shape))
}

// contiguous returns t if already row-major, else a row-major copy.
func (t *Tensor) Contiguous() *Tensor {
	if t.IsContiguous() {
		return t
	}

	out := New(t.Shape...)
	o := newOdometer(t.Shape, t)
	for i := range out.Data {
		out.Data[i] = t.Data[o.offs[0]]
		o.next()
	}
	return out
}

// numel is the number of elements a shape holds.
func numel(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

// offset maps idx to a position in data: sum of idx[i] * strides[i].
// panics on wrong index count or out of range.
func (t *Tensor) offset(idx []int) int {
	if len(idx) != len(t.Shape) {
		panic(fmt.Sprintf("tensor: got %d indices %v for %d-dimensional shape %v",
			len(idx), idx, len(t.Shape), t.Shape))
	}

	off := 0
	for i, x := range idx {
		if x < 0 || x >= t.Shape[i] {
			panic(fmt.Sprintf("tensor: index %d out of range for dim %d of shape %v",
				x, i, t.Shape))
		}
		off += x * t.Strides[i]
	}
	return off
}
