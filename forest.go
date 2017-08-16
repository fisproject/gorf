package gorf

import (
	"log"
	"math/rand"
	"sort"
	"time"
)

type Forest struct {
	K          int // k features
	estimators int
	maxDepth   int
	Trees      []*Tree
	bagger     *Bagger
}

func NewForest(task string, k, estimators, depth int) *Forest {
	f := &Forest{}
	f.K = k
	f.estimators = estimators
	f.maxDepth = depth
	f.Trees = nil
	f.bagger = NewBagger(task)
	return f
}

// Build a forest of trees from the training set.
func (f *Forest) Build(features [][]float64, labels []float64) *Forest {
	f.Trees = make([]*Tree, f.estimators)
	done := make(chan bool)

	for i := 0; i < f.estimators; i++ {
		go func(n int) {
			log.Printf("Buiding %vth tree...\n", n)
			f.Trees[n] = NewTree(f.bagger.task, f.maxDepth)
			subFeatures, subLabels := f.bagger.BootstrapSampling(features, labels)
			f.Trees[n].Build(subFeatures, subLabels)
			done <- true
		}(i)
	}

	for i := 1; i <= f.estimators; i++ {
		<-done
	}

	return f
}

// Predict class for X.
func (f *Forest) Predict(features [][]float64) (predictions []float64) {
	for _, v := range features {
		tmp := []float64{}
		for i := 0; i < f.estimators; i++ {
			tmp = append(tmp, f.Trees[i].Predict(v))
		}
		predictions = append(predictions, f.bagger.Aggregate(tmp))
	}
	return predictions
}

// Using a random selection of features.
func selectRandomFeatures(n int, k int) (selected []int) {
	rand.Seed(time.Now().UnixNano())

	tmp := make([]int, n)
	for i := 0; i < n; i++ {
		tmp[i] = i
	}
	for i := 0; i < k; i++ {
		j := i + int(rand.Float64()*float64(n-i))
		tmp[i], tmp[j] = tmp[j], tmp[i]
	}

	selected = tmp[:k]
	sort.Ints(selected)

	return selected
}
