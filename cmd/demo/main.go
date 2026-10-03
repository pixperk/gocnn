package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/pixperk/gocnn/dataset"
	"github.com/pixperk/gocnn/models"
	"github.com/pixperk/gocnn/nn"
	"github.com/pixperk/gocnn/optim"
	"github.com/pixperk/gocnn/tensor"
)

//go:embed index.html
var page []byte

const (
	numSamples   = 24
	liveEvalSize = 1000
	evalEvery    = 50
	stepEvery    = 5
)

type config struct {
	arch   string
	epochs int
	batch  int
	lr     float64
	seed   uint64
}

type server struct {
	ctx         context.Context
	cfg         config
	train, test *dataset.Dataset
	hub         *hub
	running     atomic.Bool

	mu    sync.Mutex
	model *nn.Sequential
}

func main() {
	addr := flag.String("addr", "localhost:8080", "listen address")
	dataDir := flag.String("data", "data", "directory holding the mnist .gz files")
	arch := flag.String("model", "mlp", "model to train: "+strings.Join(models.Names(), ", "))
	epochs := flag.Int("epochs", 3, "passes over the training set")
	batch := flag.Int("batch", 64, "examples per step")
	lr := flag.Float64("lr", 0.1, "learning rate")
	seed := flag.Uint64("seed", 1, "random seed")
	flag.Parse()

	model, err := models.Build(*arch, rand.New(rand.NewPCG(*seed, *seed)))
	if err != nil {
		log.Fatal(err)
	}
	train, err := dataset.Load(
		filepath.Join(*dataDir, "train-images-idx3-ubyte.gz"),
		filepath.Join(*dataDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		log.Fatal(err)
	}
	test, err := dataset.Load(
		filepath.Join(*dataDir, "t10k-images-idx3-ubyte.gz"),
		filepath.Join(*dataDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	s := &server{
		ctx:   ctx,
		cfg:   config{arch: *arch, epochs: *epochs, batch: *batch, lr: *lr, seed: *seed},
		train: train,
		test:  test,
		hub:   newHub(),
		model: model,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handlePage)
	mux.HandleFunc("GET /api/info", s.handleInfo)
	mux.HandleFunc("GET /api/events", s.handleEvents)
	mux.HandleFunc("POST /api/train", s.handleTrain)
	mux.HandleFunc("POST /api/predict", s.handlePredict)

	srv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		BaseContext:       func(net.Listener) context.Context { return ctx },
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.Shutdown(shutdown)
	}()

	log.Printf("demo running at http://%s", *addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

// handlePage serves the demo page.
func (s *server) handlePage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(page)
}

// handleInfo describes the model and sends the sample digits shown on the page.
func (s *server) handleInfo(w http.ResponseWriter, r *http.Request) {
	type sample struct {
		Label  int   `json:"label"`
		Pixels []int `json:"pixels"`
	}
	samples := make([]sample, numSamples)
	for i := range samples {
		px := make([]int, 784)
		for j := range px {
			px[j] = int(math.Round(float64(s.test.Images.Data[i*784+j]) * 255))
		}
		samples[i] = sample{Label: s.test.Labels[i], Pixels: px}
	}

	s.mu.Lock()
	params := 0
	for _, p := range s.model.Params() {
		params += len(p.Value.Data)
	}
	s.mu.Unlock()

	writeJSON(w, map[string]any{
		"arch":     models.Label(s.cfg.arch),
		"liveEval": liveEvalSize,
		"params":   params,
		"train":    s.train.Len(),
		"test":     s.test.Len(),
		"samples":  samples,
	})
}

// handleEvents streams training progress as server-sent events.
func (s *server) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")

	ch, history := s.hub.subscribe()
	defer s.hub.unsubscribe(ch)

	hello, _ := json.Marshal(map[string]any{"type": "hello", "running": s.running.Load()})
	fmt.Fprintf(w, "data: %s\n\n", hello)
	for _, b := range history {
		fmt.Fprintf(w, "data: %s\n\n", b)
	}
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return
		case b := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
	}
}

// handleTrain starts a fresh training run unless one is already going.
func (s *server) handleTrain(w http.ResponseWriter, r *http.Request) {
	if !s.running.CompareAndSwap(false, true) {
		http.Error(w, "already training", http.StatusConflict)
		return
	}
	go s.run()
	w.WriteHeader(http.StatusAccepted)
}

// handlePredict classifies one drawn 28x28 digit.
func (s *server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Pixels []float32 `json:"pixels"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	if len(req.Pixels) != 784 {
		http.Error(w, "want 784 pixels", http.StatusBadRequest)
		return
	}

	x := tensor.New(1, 784)
	for i, v := range req.Pixels {
		x.Data[i] = min(max(v, 0), 1)
	}
	s.mu.Lock()
	probs := nn.Softmax(s.model.Forward(x))
	s.mu.Unlock()

	writeJSON(w, map[string]any{"probs": probs.Data, "pred": probs.ArgMax(1)[0]})
}

// run trains a new model from scratch, publishing progress as it goes.
func (s *server) run() {
	defer s.running.Store(false)

	r := rand.New(rand.NewPCG(s.cfg.seed, s.cfg.seed))
	model, _ := models.Build(s.cfg.arch, r)
	s.mu.Lock()
	s.model = model
	s.mu.Unlock()

	params := model.Params()
	var lossFn nn.SoftmaxCrossEntropy
	opt := &optim.SGD{LR: float32(s.cfg.lr)}
	n := s.train.Len()
	stepsPerEpoch := (n + s.cfg.batch - 1) / s.cfg.batch

	s.hub.reset()
	s.hub.publish(map[string]any{"type": "start", "epochs": s.cfg.epochs, "stepsPerEpoch": stepsPerEpoch})
	s.publishEval(0)

	start := time.Now()
	step := 0
	for epoch := 1; epoch <= s.cfg.epochs; epoch++ {
		epochStart := time.Now()
		perm := r.Perm(n)
		var epochLoss, window float64
		var windowSteps, epochSteps int

		for i := 0; i < n; i += s.cfg.batch {
			if s.ctx.Err() != nil {
				return
			}
			x, y := s.train.Batch(perm[i:min(i+s.cfg.batch, n)])

			s.mu.Lock()
			loss := lossFn.Forward(model.Forward(x), y)
			model.Backward(lossFn.Backward())
			opt.Step(params)
			s.mu.Unlock()

			step++
			epochSteps++
			windowSteps++
			epochLoss += float64(loss)
			window += float64(loss)

			if step%stepEvery == 0 {
				s.hub.publish(map[string]any{
					"type":         "step",
					"step":         step,
					"epoch":        epoch,
					"loss":         window / float64(windowSteps),
					"imagesPerSec": float64(step*s.cfg.batch) / time.Since(start).Seconds(),
				})
				window, windowSteps = 0, 0
			}
			if step%evalEvery == 0 {
				s.publishEval(step)
			}
		}

		s.mu.Lock()
		acc, _ := s.evaluate(s.test.Len())
		s.mu.Unlock()
		s.hub.publish(map[string]any{
			"type":    "epoch",
			"epoch":   epoch,
			"acc":     acc,
			"loss":    epochLoss / float64(epochSteps),
			"seconds": time.Since(epochStart).Seconds(),
		})
	}
	s.hub.publish(map[string]any{"type": "done", "seconds": time.Since(start).Seconds()})
}

// publishEval sends accuracy on the live test subset and predictions for the samples.
func (s *server) publishEval(step int) {
	s.mu.Lock()
	acc, preds := s.evaluate(liveEvalSize)
	s.mu.Unlock()
	s.hub.publish(map[string]any{"type": "eval", "step": step, "acc": acc, "preds": preds[:numSamples]})
}

// evaluate returns accuracy on the first n test images and their predictions; caller holds mu.
func (s *server) evaluate(n int) (float64, []int) {
	const chunk = 1000
	preds := make([]int, 0, n)
	for start := 0; start < n; start += chunk {
		idx := make([]int, 0, chunk)
		for i := start; i < min(start+chunk, n); i++ {
			idx = append(idx, i)
		}
		x, _ := s.test.Batch(idx)
		preds = append(preds, s.model.Forward(x).ArgMax(1)...)
	}
	correct := 0
	for i, p := range preds {
		if p == s.test.Labels[i] {
			correct++
		}
	}
	return 100 * float64(correct) / float64(n), preds
}

// writeJSON writes v as a json response.
func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("write json: %v", err)
	}
}

// hub fans events out to every connected page and keeps the current run's history.
type hub struct {
	mu      sync.Mutex
	subs    map[chan []byte]struct{}
	history [][]byte
}

// newHub makes an empty hub.
func newHub() *hub {
	return &hub{subs: make(map[chan []byte]struct{})}
}

// publish records v and sends it to every subscriber that can keep up.
func (h *hub) publish(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		log.Printf("marshal event: %v", err)
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.history = append(h.history, b)
	for ch := range h.subs {
		select {
		case ch <- b:
		default:
		}
	}
}

// reset forgets the previous run's history.
func (h *hub) reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.history = nil
}

// subscribe registers a listener and returns the history so far.
func (h *hub) subscribe() (chan []byte, [][]byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	ch := make(chan []byte, 1024)
	h.subs[ch] = struct{}{}
	return ch, append([][]byte(nil), h.history...)
}

// unsubscribe removes a listener.
func (h *hub) unsubscribe(ch chan []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.subs, ch)
}
