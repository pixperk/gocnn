package dataset

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/pixperk/gocnn/tensor"
)

const (
	imagesMagic = 2051
	labelsMagic = 2049
)

// dataset holds images as (n, 784) pixels in 0..1 and their digit labels.
type Dataset struct {
	Images *tensor.Tensor
	Labels []int
}

// load reads a gzipped idx image file and its matching label file.
func Load(imagesPath, labelsPath string) (*Dataset, error) {
	ih, pixels, err := readIDX(imagesPath, imagesMagic, 4)
	if err != nil {
		return nil, err
	}
	lh, labels, err := readIDX(labelsPath, labelsMagic, 2)
	if err != nil {
		return nil, err
	}

	n, rows, cols := int(ih[1]), int(ih[2]), int(ih[3])
	if int(lh[1]) != n {
		return nil, fmt.Errorf("mnist: %d images but %d labels", n, lh[1])
	}

	d := &Dataset{Images: tensor.New(n, rows*cols), Labels: make([]int, n)}
	for i, p := range pixels {
		d.Images.Data[i] = float32(p) / 255
	}
	for i, l := range labels {
		if l > 9 {
			return nil, fmt.Errorf("mnist: %s: label %d at %d is not a digit", labelsPath, l, i)
		}
		d.Labels[i] = int(l)
	}
	return d, nil
}

// readIDX returns an idx file's header (magic first) and the bytes it describes.
func readIDX(path string, magic uint32, headerLen int) ([]uint32, []byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("mnist: %w", err)
	}
	defer f.Close()

	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, fmt.Errorf("mnist: %s: %w", path, err)
	}
	defer zr.Close()

	header := make([]uint32, headerLen)
	if err := binary.Read(zr, binary.BigEndian, header); err != nil {
		return nil, nil, fmt.Errorf("mnist: %s: header: %w", path, err)
	}
	if header[0] != magic {
		return nil, nil, fmt.Errorf("mnist: %s: magic %d, want %d", path, header[0], magic)
	}

	size := 1
	for _, d := range header[1:] {
		size *= int(d)
	}
	body := make([]byte, size)
	if _, err := io.ReadFull(zr, body); err != nil {
		return nil, nil, fmt.Errorf("mnist: %s: body: %w", path, err)
	}
	return header, body, nil
}
