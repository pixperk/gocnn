package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// equal logits mean a blind guess: p = 1/10 each, loss = ln 10.
func TestLossUniform(t *testing.T) {
	var s SoftmaxCrossEntropy
	loss := s.Forward(tensor.New(4, 10), []int{0, 3, 7, 9})

	if !near(float64(loss), math.Log(10), 1e-5) {
		t.Errorf("loss = %v, want ln 10 = %v", loss, math.Log(10))
	}
	for i, p := range s.Probs().Data {
		if !near(float64(p), 0.1, 1e-6) {
			t.Fatalf("prob[%d] = %v, want 0.1", i, p)
		}
	}
}

// logits [0, ln 3] give probs [1/4, 3/4].
func TestLossByHand(t *testing.T) {
	var s SoftmaxCrossEntropy
	logits := from([]float32{0, float32(math.Log(3))}, 1, 2)

	loss := s.Forward(logits, []int{1})

	if p := s.Probs().Data; !near(float64(p[0]), 0.25, 1e-6) || !near(float64(p[1]), 0.75, 1e-6) {
		t.Errorf("probs = %v, want [0.25 0.75]", p)
	}
	if want := -math.Log(0.75); !near(float64(loss), want, 1e-6) {
		t.Errorf("loss = %v, want -ln 0.75 = %v", loss, want)
	}
}

func TestLossProbsSumToOne(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	logits := tensor.New(5, 10)
	for i := range logits.Data {
		logits.Data[i] = r.Float32()*20 - 10
	}
	var s SoftmaxCrossEntropy
	s.Forward(logits, []int{0, 1, 2, 3, 4})

	sums := s.Probs().Sum(1, false)
	for i, v := range sums.Data {
		if !near(float64(v), 1, 1e-5) {
			t.Errorf("row %d sums to %v, want 1", i, v)
		}
	}
}

// huge logits overflow exp without the max shift.
func TestLossIsStable(t *testing.T) {
	var s SoftmaxCrossEntropy
	logits := from([]float32{1000, 0, 0, 1000}, 2, 2)

	loss := s.Forward(logits, []int{0, 0})

	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss = %v, want a finite number", loss)
	}
	// row 0 right with full confidence (~0), row 1 wrong by 1000 (~1000).
	if !near(float64(loss), 500, 1e-3) {
		t.Errorf("loss = %v, want ~500", loss)
	}
}

func TestLossBackwardByHand(t *testing.T) {
	var s SoftmaxCrossEntropy
	s.Forward(from([]float32{0, float32(math.Log(3)), 0, float32(math.Log(3))}, 2, 2), []int{1, 0})

	// probs [0.25 0.75] per row; minus onehot; divided by n = 2.
	want := []float64{0.125, -0.125, -0.375, 0.375}
	for i, g := range s.Backward().Data {
		if !near(float64(g), want[i], 1e-6) {
			t.Errorf("grad = %v, want %v", s.Backward().Data, want)
			break
		}
	}
}

func TestLossGradCheck(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	logits := tensor.New(4, 5)
	for i := range logits.Data {
		logits.Data[i] = r.Float32()*4 - 2
	}
	labels := []int{2, 0, 4, 1}

	var s SoftmaxCrossEntropy
	s.Forward(logits, labels)
	grad := s.Backward()

	const h = 1e-2
	for i := range logits.Data {
		orig := logits.Data[i]
		logits.Data[i] = orig + h
		lp := float64(s.Forward(logits, labels))
		logits.Data[i] = orig - h
		lm := float64(s.Forward(logits, labels))
		logits.Data[i] = orig

		numeric := (lp - lm) / (2 * h)
		if !near(float64(grad.Data[i]), numeric, 1e-3) {
			t.Errorf("grad[%d] = %.5f, finite difference = %.5f", i, grad.Data[i], numeric)
		}
	}
}

func TestLossPanics(t *testing.T) {
	tests := []struct {
		name   string
		logits *tensor.Tensor
		labels []int
	}{
		{"label too big", tensor.New(2, 3), []int{0, 3}},
		{"negative label", tensor.New(2, 3), []int{-1, 0}},
		{"label count mismatch", tensor.New(2, 3), []int{0}},
		{"logits not 2d", tensor.New(6), []int{0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", tt.name)
				}
			}()
			var s SoftmaxCrossEntropy
			s.Forward(tt.logits, tt.labels)
		})
	}
}
