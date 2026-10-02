package tensor

import "fmt"

type Tensor struct {
	//data is just a slice represention tensor's data
	Data []float32
	//example : shape = [2,3] means 2 rows and 3 columns
	Shape []int
}

// new tensor with some shape
func New(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}

	return &Tensor{
		Data:  make([]float32, size),
		Shape: append([]int(nil), shape...),
	}
}

// at returns the element at idx, one index per dim.
// row-major: for shape [2,3,4], at(i,j,k) reads data[i*12 + j*4 + k].
// panics on wrong index count or out of range.
func (t *Tensor) At(idx ...int) float32 {
	if len(idx) != len(t.Shape) {
		panic(fmt.Sprintf("tensor: At got %d indices %v for %d-dimensional shape %v",
			len(idx), idx, len(t.Shape), t.Shape))
	}

	offset := 0
	step := 1 // elements to skip per step along dim i
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if idx[i] < 0 || idx[i] >= t.Shape[i] {
			panic(fmt.Sprintf("tensor: index %d out of range for dim %d of shape %v",
				idx[i], i, t.Shape))
		}
		offset += idx[i] * step
		step *= t.Shape[i]
	}
	return t.Data[offset]
}
