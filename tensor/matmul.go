package tensor

import "fmt"

// matmul multiplies a (m,k) by b (k,n) into a new (m,n) tensor.
func MatMul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 || a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("tensor: cannot matmul %v by %v", a.Shape, b.Shape))
	}

	a, b = a.Contiguous(), b.Contiguous()
	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]

	out := New(m, n)
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a.Data[i*k+p] * b.Data[p*n+j]
			}
			out.Data[i*n+j] = sum
		}
	}
	return out
}
