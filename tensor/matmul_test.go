package tensor

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// matmulRef is the slow, obviously correct oracle for every matmul.
func matmulRef(a, b *Tensor) *Tensor {
	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	out := New(m, n)
	for i := range m {
		for j := range n {
			var s float32
			for p := range k {
				s += a.At(i, p) * b.At(p, j)
			}
			out.Set(s, i, j)
		}
	}
	return out
}

func randTensor(r *rand.Rand, shape ...int) *Tensor {
	x := New(shape...)
	for i := range x.Data {
		x.Data[i] = r.Float32()*2 - 1
	}
	return x
}

func assertClose(t *testing.T, got, want *Tensor, tol float32) {
	t.Helper()
	if !slices.Equal(got.Shape, want.Shape) {
		t.Fatalf("Shape = %v, want %v", got.Shape, want.Shape)
	}
	g, w := got.Contiguous().Data, want.Contiguous().Data
	for i := range g {
		if d := g[i] - w[i]; d > tol || d < -tol {
			t.Fatalf("element %d = %v, want %v", i, g[i], w[i])
		}
	}
}

func TestMatMulByHand(t *testing.T) {
	a := from([]float32{
		2, 1, 3,
		4, 0, 2,
	}, 2, 3)
	b := from([]float32{
		5, 2,
		1, 4,
		3, 7,
	}, 3, 2)

	want := from([]float32{
		20, 29,
		26, 22,
	}, 2, 2)
	assertClose(t, MatMul(a, b), want, 0)
}

func TestMatMulIdentity(t *testing.T) {
	a := fill(New(3, 3))
	id := from([]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3)

	assertClose(t, MatMul(a, id), a, 0)
	assertClose(t, MatMul(id, a), a, 0)
}

func TestMatMulShapes(t *testing.T) {
	tests := []struct {
		a, b, want []int
	}{
		{[]int{2, 3}, []int{3, 4}, []int{2, 4}},
		{[]int{1, 5}, []int{5, 1}, []int{1, 1}}, // row times column: dot product
		{[]int{5, 1}, []int{1, 5}, []int{5, 5}}, // column times row: outer product
		{[]int{32, 784}, []int{784, 10}, []int{32, 10}},
	}
	for _, tt := range tests {
		got := MatMul(New(tt.a...), New(tt.b...))
		if !slices.Equal(got.Shape, tt.want) {
			t.Errorf("%v x %v: Shape = %v, want %v", tt.a, tt.b, got.Shape, tt.want)
		}
	}
}

func TestMatMulRandomAgainstRef(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	sizes := [][3]int{{1, 1, 1}, {3, 5, 2}, {7, 4, 9}, {16, 16, 16}, {33, 17, 20}}
	for _, s := range sizes {
		a, b := randTensor(r, s[0], s[1]), randTensor(r, s[1], s[2])
		assertClose(t, MatMul(a, b), matmulRef(a, b), 1e-5)
	}
}

// inputs may be views with odd strides; result must follow logical order.
func TestMatMulTransposedInputs(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	a := randTensor(r, 4, 3).Transpose(0, 1) // logical [3,4]
	b := randTensor(r, 5, 4).Transpose(0, 1) // logical [4,5]

	assertClose(t, MatMul(a, b), matmulRef(a, b), 1e-5)
}

func TestMatMulResultIsFresh(t *testing.T) {
	a := from([]float32{1, 2, 3, 4}, 2, 2)
	b := from([]float32{5, 6, 7, 8}, 2, 2)

	got := MatMul(a, b)

	if !got.IsContiguous() {
		t.Errorf("result not contiguous: strides %v", got.Strides)
	}
	if !slices.Equal(a.Data, []float32{1, 2, 3, 4}) || !slices.Equal(b.Data, []float32{5, 6, 7, 8}) {
		t.Errorf("inputs changed: a %v, b %v", a.Data, b.Data)
	}
}

func TestMatMulPanics(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
	}{
		{"inner dims differ", []int{2, 3}, []int{4, 2}},
		{"a is 1d", []int{3}, []int{3, 2}},
		{"b is 3d", []int{2, 3}, []int{2, 3, 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("MatMul(%v, %v) did not panic", tt.a, tt.b)
				}
			}()
			MatMul(New(tt.a...), New(tt.b...))
		})
	}
}
