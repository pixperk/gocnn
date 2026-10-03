package models

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/pixperk/gocnn/tensor"
)

// every model maps a batch of flat mnist images to 10 scores each.
func TestModelsMapImagesToTenScores(t *testing.T) {
	for _, name := range Names() {
		t.Run(name, func(t *testing.T) {
			m, err := Build(name, rand.New(rand.NewPCG(1, 2)))
			if err != nil {
				t.Fatal(err)
			}
			y := m.Forward(tensor.New(3, 784))
			if !slices.Equal(y.Shape, []int{3, 10}) {
				t.Errorf("output shape = %v, want [3 10]", y.Shape)
			}
			if dx := m.Backward(tensor.New(3, 10)); !slices.Equal(dx.Shape, []int{3, 784}) {
				t.Errorf("dx shape = %v, want [3 784]", dx.Shape)
			}
			if Label(name) == "" {
				t.Error("missing label")
			}
		})
	}
}

func TestBuildUnknown(t *testing.T) {
	if _, err := Build("transformer", rand.New(rand.NewPCG(1, 2))); err == nil {
		t.Error("Build of an unknown model returned no error")
	}
}

func TestNames(t *testing.T) {
	if got := Names(); !slices.Equal(got, []string{"cnn", "mlp"}) {
		t.Errorf("Names() = %v, want [cnn mlp]", got)
	}
}
