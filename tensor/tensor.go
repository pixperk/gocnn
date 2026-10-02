package tensor

import "fmt"

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
	size := 1
	for _, s := range shape {
		size *= s
	}

	return &Tensor{
		Data:    make([]float32, size),
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
