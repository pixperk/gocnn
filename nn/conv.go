package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/pixperk/gocnn/tensor"
)

// conv2d slides outC learned (inC, k, k) filters over an (n, inC, h, w) tensor.
type Conv2D struct {
	W, B        *Param
	Stride, Pad int
	x           *tensor.Tensor
}

// newconv2d makes a conv layer with he-initialised filters.
func NewConv2D(inC, outC, k, stride, pad int, r *rand.Rand) *Conv2D {
	std := math.Sqrt(2 / float64(inC*k*k))
	w := tensor.New(outC, inC, k, k)
	for i := range w.Data {
		w.Data[i] = float32(r.NormFloat64() * std)
	}
	return &Conv2D{W: newParam(w), B: newParam(tensor.New(outC)), Stride: stride, Pad: pad}
}

// forward caches x and returns the filter responses at every position.
func (c *Conv2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	outC, inC, k := c.W.Value.Shape[0], c.W.Value.Shape[1], c.W.Value.Shape[2]
	if len(x.Shape) != 4 || x.Shape[1] != inC {
		panic(fmt.Sprintf("nn: conv wants (n, %d, h, w), got %v", inC, x.Shape))
	}
	x = x.Contiguous()
	c.x = x
	n, h, w := x.Shape[0], x.Shape[2], x.Shape[3]
	oh, ow := c.outSize(h, w)

	out := tensor.New(n, outC, oh, ow)
	xd, wd, bd := x.Data, c.W.Value.Data, c.B.Value.Data
	o := 0
	for b := range n {
		for oc := range outC {
			for i := range oh {
				for j := range ow {
					sum := bd[oc]
					for ic := range inC {
						for ki := range k {
							ih := i*c.Stride + ki - c.Pad
							if ih < 0 || ih >= h {
								continue
							}
							for kj := range k {
								iw := j*c.Stride + kj - c.Pad
								if iw < 0 || iw >= w {
									continue
								}
								sum += xd[((b*inC+ic)*h+ih)*w+iw] * wd[((oc*inC+ic)*k+ki)*k+kj]
							}
						}
					}
					out.Data[o] = sum
					o++
				}
			}
		}
	}
	return out
}

// backward runs forward's loops again, turning each sum += x·w into dW += x·g and dx += w·g.
func (c *Conv2D) Backward(dy *tensor.Tensor) *tensor.Tensor {
	outC, inC, k := c.W.Value.Shape[0], c.W.Value.Shape[1], c.W.Value.Shape[2]
	n, h, w := c.x.Shape[0], c.x.Shape[2], c.x.Shape[3]
	oh, ow := c.outSize(h, w)
	dy = dy.Contiguous()

	dx := tensor.New(c.x.Shape...)
	xd, wd := c.x.Data, c.W.Value.Data
	dwd, dbd := c.W.Grad.Data, c.B.Grad.Data
	o := 0
	for b := range n {
		for oc := range outC {
			for i := range oh {
				for j := range ow {
					g := dy.Data[o]
					o++
					dbd[oc] += g
					for ic := range inC {
						for ki := range k {
							ih := i*c.Stride + ki - c.Pad
							if ih < 0 || ih >= h {
								continue
							}
							for kj := range k {
								iw := j*c.Stride + kj - c.Pad
								if iw < 0 || iw >= w {
									continue
								}
								xi := ((b*inC+ic)*h+ih)*w + iw
								wi := ((oc*inC+ic)*k+ki)*k + kj
								dwd[wi] += xd[xi] * g
								dx.Data[xi] += wd[wi] * g
							}
						}
					}
				}
			}
		}
	}
	return dx
}

// params returns the filters and biases.
func (c *Conv2D) Params() []*Param {
	return []*Param{c.W, c.B}
}

// outSize is the output height and width for an h×w input.
func (c *Conv2D) outSize(h, w int) (int, int) {
	k := c.W.Value.Shape[2]
	oh := (h+2*c.Pad-k)/c.Stride + 1
	ow := (w+2*c.Pad-k)/c.Stride + 1
	if oh < 1 || ow < 1 {
		panic(fmt.Sprintf("nn: conv kernel %d does not fit a %dx%d input with pad %d", k, h, w, c.Pad))
	}
	return oh, ow
}
