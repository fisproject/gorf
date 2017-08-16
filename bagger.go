package gorf

import (
	"math/rand"
	"time"
)

// Bagging is a parallel ensemble methods.
type Bagger struct {
	task string
}

func NewBagger(task string) *Bagger {
	b := &Bagger{}
	b.task = task
	return b
}

// Get a subset to train the base learner.
func (b *Bagger) BootstrapSampling(features [][]float64, labels []float64) (subFeatures [][]float64, subLabels []float64) {
	rand.Seed(time.Now().UnixNano())
	n := len(features)
	rnd := []int{}

	// restoration sampling
	for i := 0; i < n; i++ {
		rnd = append(rnd, rand.Intn(n))
	}

	for _, v := range rnd {
		subFeatures = append(subFeatures, features[v])
		subLabels = append(subLabels, labels[v])
	}

	return subFeatures, subLabels
}

// Aggregate the output of the base learner.
func (b *Bagger) Aggregate(predictions []float64) float64 {
	switch b.task {
	case "regression", "mse":
		return b.average(predictions)
	case "classification", "gini", "entropy":
		return b.vote(predictions)
	default:
		return b.vote(predictions)
	}
}

// Voting for classification.
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

// Averaging for regression.
func (b *Bagger) average(predictions []float64) float64 {
	sum := 0.0
	for _, v := range predictions {
		sum += v
	}
	return sum / float64(len(predictions))
}
