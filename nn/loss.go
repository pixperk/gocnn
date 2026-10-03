package nn

import (
	"fmt"
	"math"

	"github.com/pixperk/gocnn/tensor"
)

// softmaxCrossEntropy turns logits into probabilities and scores them against labels.
type SoftmaxCrossEntropy struct {
	probs  *tensor.Tensor
	labels []int
}

// forward returns the mean loss over the batch for logits (n, classes).
func (s *SoftmaxCrossEntropy) Forward(logits *tensor.Tensor, labels []int) float32 {
	if len(logits.Shape) != 2 || logits.Shape[0] != len(labels) {
		panic(fmt.Sprintf("nn: loss got logits %v for %d labels", logits.Shape, len(labels)))
	}
	n, classes := logits.Shape[0], logits.Shape[1]

	shifted := tensor.Sub(logits, logits.Max(1, true))
	exps := shifted.Apply(func(v float32) float32 { return float32(math.Exp(float64(v))) })
	sums := exps.Sum(1, true)
	s.probs = tensor.Div(exps, sums)
	s.labels = labels

	var total float64
	for i, y := range labels {
		if y < 0 || y >= classes {
			panic(fmt.Sprintf("nn: label %d out of range for %d classes", y, classes))
		}
		total += math.Log(float64(sums.Data[i])) - float64(shifted.Data[i*classes+y])
	}
	return float32(total / float64(n))
}

// backward returns dloss/dlogits = (probs - onehot) / n.
func (s *SoftmaxCrossEntropy) Backward() *tensor.Tensor {
	n, classes := s.probs.Shape[0], s.probs.Shape[1]
	grad := s.probs.Apply(func(p float32) float32 { return p / float32(n) })
	for i, y := range s.labels {
		grad.Data[i*classes+y] -= 1 / float32(n)
	}
	return grad
}

// softmax turns each row of logits (n, classes) into probabilities.
func Softmax(logits *tensor.Tensor) *tensor.Tensor {
	shifted := tensor.Sub(logits, logits.Max(1, true))
	exps := shifted.Apply(func(v float32) float32 { return float32(math.Exp(float64(v))) })
	return tensor.Div(exps, exps.Sum(1, true))
}

// probs returns the softmax probabilities from the last forward.
func (s *SoftmaxCrossEntropy) Probs() *tensor.Tensor {
	return s.probs
}
