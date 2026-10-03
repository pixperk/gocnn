package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

// gradCheck compares a layer's backward against finite differences of loss = sum(y * r).
func gradCheck(t *testing.T, l Layer, x *tensor.Tensor, r *rand.Rand) {
	t.Helper()
	const h = 1e-2

	y := l.Forward(x)
	rw := tensor.New(y.Shape...)
	for i := range rw.Data {
		rw.Data[i] = r.Float32()*2 - 1
	}

	loss := func() float64 {
		y := l.Forward(x).Contiguous()
		var s float64
		for i, v := range y.Data {
			s += float64(v) * float64(rw.Data[i])
		}
		return s
	}

	for _, p := range l.Params() {
		clear(p.Grad.Data)
	}
	l.Forward(x)
	dx := l.Backward(rw).Contiguous()

	check := func(name string, data []float32, analytic []float32) {
		for i := range data {
			orig := data[i]
			data[i] = orig + h
			lp := loss()
			data[i] = orig - h
			lm := loss()
			data[i] = orig

			numeric := (lp - lm) / (2 * h)
			a := float64(analytic[i])
			if math.Abs(a-numeric) > 1e-2*math.Max(1, math.Abs(a)+math.Abs(numeric)) {
				t.Errorf("%s[%d]: backward says %.5f, finite difference says %.5f", name, i, a, numeric)
				return
			}
		}
	}

	for k, p := range l.Params() {
		check(fmt.Sprintf("param %d", k), p.Value.Data, p.Grad.Data)
	}
	check("x", x.Data, dx.Data)
}
