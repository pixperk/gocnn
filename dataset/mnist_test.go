package dataset

import (
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// writeIDX writes a gzipped idx file: big-endian uint32 header, then raw bytes.
func writeIDX(t *testing.T, header []uint32, body []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "f.gz")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	zw := gzip.NewWriter(f)
	if err := binary.Write(zw, binary.BigEndian, header); err != nil {
		t.Fatal(err)
	}
	if _, err := zw.Write(body); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// two 2x2 images and their labels.
func tinyFiles(t *testing.T) (images, labels string) {
	images = writeIDX(t, []uint32{2051, 2, 2, 2}, []byte{0, 255, 51, 102, 255, 0, 0, 255})
	labels = writeIDX(t, []uint32{2049, 2}, []byte{7, 3})
	return images, labels
}

func TestLoadTiny(t *testing.T) {
	d, err := Load(tinyFiles(t))
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(d.Images.Shape, []int{2, 4}) {
		t.Errorf("Images shape = %v, want [2 4]", d.Images.Shape)
	}
	if want := []float32{0, 1, 0.2, 0.4, 1, 0, 0, 1}; !slices.Equal(d.Images.Data, want) {
		t.Errorf("Images = %v, want %v", d.Images.Data, want)
	}
	if want := []int{7, 3}; !slices.Equal(d.Labels, want) {
		t.Errorf("Labels = %v, want %v", d.Labels, want)
	}
}

func TestLoadErrors(t *testing.T) {
	goodImages, goodLabels := tinyFiles(t)
	notGzip := filepath.Join(t.TempDir(), "plain")
	if err := os.WriteFile(notGzip, []byte("not gzip"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name           string
		images, labels string
	}{
		{"missing images file", "nope.gz", goodLabels},
		{"missing labels file", goodImages, "nope.gz"},
		{"labels passed as images", goodLabels, goodLabels},
		{"images passed as labels", goodImages, goodImages},
		{"not gzipped", notGzip, goodLabels},
		{"empty header", writeIDX(t, nil, nil), goodLabels},
		{"truncated pixels", writeIDX(t, []uint32{2051, 2, 2, 2}, []byte{1, 2, 3}), goodLabels},
		{"truncated labels", goodImages, writeIDX(t, []uint32{2049, 2}, []byte{7})},
		{"count mismatch", goodImages, writeIDX(t, []uint32{2049, 3}, []byte{1, 2, 3})},
		{"label above 9", goodImages, writeIDX(t, []uint32{2049, 2}, []byte{7, 10})},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := Load(tt.images, tt.labels); err == nil {
				t.Error("Load returned no error")
			}
		})
	}
}

// runs only after the real dataset is downloaded into data/.
func TestLoadRealMNIST(t *testing.T) {
	images := filepath.Join("..", "data", "train-images-idx3-ubyte.gz")
	labels := filepath.Join("..", "data", "train-labels-idx1-ubyte.gz")
	if _, err := os.Stat(images); err != nil {
		t.Skip("mnist not downloaded")
	}

	d, err := Load(images, labels)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(d.Images.Shape, []int{60000, 784}) {
		t.Errorf("Images shape = %v, want [60000 784]", d.Images.Shape)
	}
	if want := []int{5, 0, 4, 1, 9, 2, 1, 3}; !slices.Equal(d.Labels[:8], want) {
		t.Errorf("first labels = %v, want %v", d.Labels[:8], want)
	}
	lo, hi := slices.Min(d.Images.Data), slices.Max(d.Images.Data)
	if lo != 0 || hi != 1 {
		t.Errorf("pixel range = [%v, %v], want [0, 1]", lo, hi)
	}
}
