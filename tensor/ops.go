package tensor

import (
	"fmt"
	"slices"
)

// broadcastShapes returns the shape of a op b under numpy rules.
func broadcastShapes(a, b []int) []int {
	n := max(len(a), len(b))
	out := make([]int, n)

	for i := range n {
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

// dimFromRight returns shape's i-th dim from the right, or 1 if missing.
func dimFromRight(shape []int, i int) int {
	if i >= len(shape) {
		return 1
	}
	return shape[len(shape)-1-i]
}

// expand returns a view of t stretched to shape using stride 0.
func expand(t *Tensor, shape []int) *Tensor {
	if len(t.Shape) > len(shape) {
		panic(fmt.Sprintf("tensor: cannot expand %v to fewer dims %v", t.Shape, shape))
	}

	strides := make([]int, len(shape))
	for i := range shape {
		out := len(shape) - 1 - i
		src := len(t.Shape) - 1 - i

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

// zipWith combines a and b elementwise with f, broadcasting as needed.
func zipWith(a, b *Tensor, f func(x, y float32) float32) *Tensor {
	shape := broadcastShapes(a.Shape, b.Shape)
	ea, eb := expand(a, shape), expand(b, shape)

	out := New(shape...)
	o := newOdometer(shape, ea, eb)
	for i := range out.Data {
		out.Data[i] = f(ea.Data[o.offs[0]], eb.Data[o.offs[1]])
		o.next()
	}
	return out
}

func Add(a, b *Tensor) *Tensor { return zipWith(a, b, func(x, y float32) float32 { return x + y }) }
func Sub(a, b *Tensor) *Tensor { return zipWith(a, b, func(x, y float32) float32 { return x - y }) }
func Mul(a, b *Tensor) *Tensor { return zipWith(a, b, func(x, y float32) float32 { return x * y }) }
func Div(a, b *Tensor) *Tensor { return zipWith(a, b, func(x, y float32) float32 { return x / y }) }

// apply returns a new tensor with f applied to every element.
func (t *Tensor) Apply(f func(float32) float32) *Tensor {
	out := New(t.Shape...)
	o := newOdometer(t.Shape, t)
	for i := range out.Data {
		out.Data[i] = f(t.Data[o.offs[0]])
		o.next()
	}
	return out
}
