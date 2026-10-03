package nn

import (
	"fmt"
	"math"
	"slices"

	"github.com/pixperk/gocnn/tensor"
)

// maxpool2d keeps the largest value in each k×k window of an (n, c, h, w) tensor.
type MaxPool2D struct {
	K      int
	shape  []int
	argmax []int
}

// newmaxpool2d pools with window size and stride k.
func NewMaxPool2D(k int) *MaxPool2D {
	return &MaxPool2D{K: k}
}

// forward takes each window's max and remembers where it was.
func (m *MaxPool2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic(fmt.Sprintf("nn: maxpool wants (n, c, h, w), got %v", x.Shape))
	}
	x = x.Contiguous()
	n, c, h, w := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	oh, ow := h/m.K, w/m.K

	out := tensor.New(n, c, oh, ow)
	m.shape = slices.Clone(x.Shape)
	m.argmax = make([]int, len(out.Data))

	o := 0
	for plane := range n * c {
		base := plane * h * w
		for i := range oh {
			for j := range ow {
				best, bestIdx := float32(math.Inf(-1)), 0
				for di := range m.K {
					for dj := range m.K {
						idx := base + (i*m.K+di)*w + j*m.K + dj
						if x.Data[idx] > best {
							best, bestIdx = x.Data[idx], idx
						}
					}
				}
				out.Data[o] = best
				m.argmax[o] = bestIdx
				o++
			}
		}
	}
	return out
}

// backward sends each output's gradient to its window's winner.
func (m *MaxPool2D) Backward(dy *tensor.Tensor) *tensor.Tensor {
	dy = dy.Contiguous()
	dx := tensor.New(m.shape...)
	for o, idx := range m.argmax {
		dx.Data[idx] += dy.Data[o]
	}
	return dx
}

// params returns nothing; maxpool has no weights.
func (m *MaxPool2D) Params() []*Param {
	return nil
}
