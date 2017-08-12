package randomforest

import (
	"math/rand"
	"time"
)

type Bagger struct {
	task string
}

func NewBagger(task string) *Bagger {
	bagger := &Bagger{}
	bagger.task = task
	return bagger
}

func (b *Bagger) BootstrapSampling(features [][]float64, labels []float64) (subsamplesF [][]float64, subsamplesL []float64) {
	rand.Seed(time.Now().UnixNano())
	n := len(features)
	rnd := []int{}

	// restoration sampling
	for i := 0; i < n; i++ {
		rnd = append(rnd, rand.Intn(n))
	}

	for _, v := range rnd {
		subsamplesF = append(subsamplesF, features[v])
		subsamplesL = append(subsamplesL, labels[v])
	}

	return subsamplesF, subsamplesL
}

func (b *Bagger) Aggregate(predictions []float64) float64 {
	switch b.task {
	case "regression", "r":
		return b.average(predictions)
	case "classification", "c":
		return b.vote(predictions)
	default:
		return b.vote(predictions)
	}
}

func (b *Bagger) vote(predictions []float64) (prediction float64) {
	uniq := unique(predictions)
	cnt := 0
	for i := 0; i < len(uniq); i++ {
		if cnt < count(predictions, uniq[i]) {
			cnt = count(predictions, uniq[i])
			prediction = uniq[i]
		}
	}
	return prediction
}

func (b *Bagger) average(predictions []float64) float64 {
	sum := 0.0
	for _, v := range predictions {
		sum += v
	}
	return sum / float64(len(predictions))
}
