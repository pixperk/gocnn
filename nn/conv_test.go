package nn

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

// one 3x3 channel, one 2x2 filter [[1 2] [3 4]], no padding.
func tinyConv() (*Conv2D, *tensor.Tensor) {
	c := NewConv2D(1, 1, 2, 1, 0, rand.New(rand.NewPCG(1, 2)))
	copy(c.W.Value.Data, []float32{1, 2, 3, 4})
	x := from([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 1, 1, 3, 3)
	return c, x
}

func TestConvForwardByHand(t *testing.T) {
	c, x := tinyConv()
	y := c.Forward(x)

	// (0,0) = 1·1 + 2·2 + 4·3 + 5·4 = 37
	if !slices.Equal(y.Shape, []int{1, 1, 2, 2}) {
		t.Fatalf("Shape = %v, want [1 1 2 2]", y.Shape)
	}
	if want := []float32{37, 47, 67, 77}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

func TestConvBias(t *testing.T) {
	c, x := tinyConv()
	c.B.Value.Data[0] = 100

	if want := []float32{137, 147, 167, 177}; !slices.Equal(c.Forward(x).Data, want) {
		t.Errorf("Data = %v, want %v", c.Forward(x).Data, want)
	}
}

// with padding 1 the border reads zeros: top-left only sees x(0,0) under w(1,1).
func TestConvPadding(t *testing.T) {
	c, x := tinyConv()
	c.Pad = 1
	y := c.Forward(x)

	if !slices.Equal(y.Shape, []int{1, 1, 4, 4}) {
		t.Fatalf("Shape = %v, want [1 1 4 4]", y.Shape)
	}
	if y.Data[0] != 4 {
		t.Errorf("top-left = %v, want 1·4 = 4", y.Data[0])
	}
	if y.Data[15] != 9 {
		t.Errorf("bottom-right = %v, want 9·1 = 9", y.Data[15])
	}
}

func TestConvStride(t *testing.T) {
	c := NewConv2D(1, 1, 2, 2, 0, rand.New(rand.NewPCG(1, 2)))
	copy(c.W.Value.Data, []float32{1, 1, 1, 1})
	x := tensor.New(1, 1, 4, 4)
	for i := range x.Data {
		x.Data[i] = float32(i)
	}

	y := c.Forward(x)

	if !slices.Equal(y.Shape, []int{1, 1, 2, 2}) {
		t.Fatalf("Shape = %v, want [1 1 2 2]", y.Shape)
	}
	if want := []float32{10, 18, 42, 50}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

// a filter sums over every input channel; each filter makes its own output channel.
func TestConvChannels(t *testing.T) {
	c := NewConv2D(2, 2, 1, 1, 0, rand.New(rand.NewPCG(1, 2)))
	copy(c.W.Value.Data, []float32{
		1, 10, // filter 0: 1·ch0 + 10·ch1
		-1, 0, // filter 1: -ch0
	})
	x := from([]float32{
		1, 2, // ch0
		3, 4, // ch1
	}, 1, 2, 1, 2)

	y := c.Forward(x)

	if want := []float32{31, 42, -1, -2}; !slices.Equal(y.Data, want) {
		t.Errorf("Data = %v, want %v", y.Data, want)
	}
}

func TestConvMNISTShapes(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	y := NewConv2D(1, 8, 3, 1, 1, r).Forward(tensor.New(2, 1, 28, 28))
	if !slices.Equal(y.Shape, []int{2, 8, 28, 28}) {
		t.Errorf("conv1 Shape = %v, want [2 8 28 28]", y.Shape)
	}
	y = NewConv2D(8, 16, 3, 1, 1, r).Forward(tensor.New(2, 8, 14, 14))
	if !slices.Equal(y.Shape, []int{2, 16, 14, 14}) {
		t.Errorf("conv2 Shape = %v, want [2 16 14 14]", y.Shape)
	}
}

func TestConvBackwardByHand(t *testing.T) {
	c, x := tinyConv()
	c.Forward(x)
	dx := c.Backward(from([]float32{1, 1, 1, 1}, 1, 1, 2, 2))

	// each weight sums the inputs it touched: w(0,0) saw 1,2,4,5.
	if want := []float32{12, 16, 24, 28}; !slices.Equal(c.W.Grad.Data, want) {
		t.Errorf("dW = %v, want %v", c.W.Grad.Data, want)
	}
	if want := []float32{4}; !slices.Equal(c.B.Grad.Data, want) {
		t.Errorf("db = %v, want %v", c.B.Grad.Data, want)
	}
	// each pixel sums the weights that touched it: the centre was under all four.
	want := []float32{
		1, 3, 2,
		4, 10, 6,
		3, 7, 4,
	}
	if !slices.Equal(dx.Data, want) {
		t.Errorf("dx = %v, want %v", dx.Data, want)
	}
}

func TestConvBackwardAccumulates(t *testing.T) {
	c, x := tinyConv()
	dy := from([]float32{1, 1, 1, 1}, 1, 1, 2, 2)
	c.Forward(x)
	c.Backward(dy)
	c.Backward(dy)

	if want := []float32{24, 32, 48, 56}; !slices.Equal(c.W.Grad.Data, want) {
		t.Errorf("dW after two backwards = %v, want %v", c.W.Grad.Data, want)
	}
}

func TestNewConvInit(t *testing.T) {
	c := NewConv2D(8, 64, 3, 1, 1, rand.New(rand.NewPCG(1, 2)))

	if !slices.Equal(c.W.Value.Shape, []int{64, 8, 3, 3}) || !slices.Equal(c.B.Value.Shape, []int{64}) {
		t.Fatalf("W %v, B %v, want [64 8 3 3] and [64]", c.W.Value.Shape, c.B.Value.Shape)
	}
	var sumSq float64
	for _, v := range c.W.Value.Data {
		sumSq += float64(v) * float64(v)
	}
	std := math.Sqrt(sumSq / float64(len(c.W.Value.Data)))
	if want := math.Sqrt(2.0 / (8 * 3 * 3)); math.Abs(std-want)/want > 0.05 {
		t.Errorf("W std = %.4f, want ~%.4f", std, want)
	}
}

func TestConvParams(t *testing.T) {
	c := NewConv2D(1, 2, 3, 1, 1, rand.New(rand.NewPCG(1, 2)))
	if ps := c.Params(); len(ps) != 2 || ps[0] != c.W || ps[1] != c.B {
		t.Errorf("Params() = %v, want [W B]", ps)
	}
}

func TestConvIsLayer(t *testing.T) {
	var _ Layer = NewConv2D(1, 1, 1, 1, 0, rand.New(rand.NewPCG(1, 2)))
}

func TestConvPanics(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	tests := []struct {
		name string
		c    *Conv2D
		x    *tensor.Tensor
	}{
		{"not 4d", NewConv2D(1, 1, 3, 1, 1, r), tensor.New(28, 28)},
		{"wrong channels", NewConv2D(3, 1, 3, 1, 1, r), tensor.New(1, 1, 28, 28)},
		{"kernel bigger than input", NewConv2D(1, 1, 5, 1, 0, r), tensor.New(1, 1, 3, 3)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", tt.name)
				}
			}()
			tt.c.Forward(tt.x)
		})
	}
}

func TestConvGradCheck(t *testing.T) {
	tests := []struct {
		name                     string
		inC, outC, k, stride, pd int
		h, w                     int
	}{
		{"same padding", 2, 3, 3, 1, 1, 5, 5},
		{"no padding", 2, 2, 3, 1, 0, 5, 4},
		{"stride 2", 1, 2, 2, 2, 0, 4, 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(13, 14))
			c := NewConv2D(tt.inC, tt.outC, tt.k, tt.stride, tt.pd, r)
			x := tensor.New(2, tt.inC, tt.h, tt.w)
			for i := range x.Data {
				x.Data[i] = r.Float32()*2 - 1
			}
			gradCheck(t, c, x, r)
		})
	}
}
