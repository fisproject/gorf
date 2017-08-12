package randomforest

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
			f.Trees[n] = NewTree(gini, f.maxDepth)
			subsamplesF, subsamplesL := f.bagger.BootstrapSampling(features, labels)
			f.Trees[n].Build(subsamplesF, subsamplesL)
			done <- true
		}(i)
	}

	for i := 1; i <= f.estimators; i++ {
		<-done
	}

	return f
}

// Predict class for X.
func (f *Forest) Predict(feature []float64) float64 {
	predictions := []float64{}
	for i := 0; i < f.estimators; i++ {
		predictions = append(predictions, f.Trees[i].Predict(feature))
	}
	log.Println(predictions)
	return f.bagger.Aggregate(predictions)
}

// For each tree randomly select feature.
func selectRandomFeatures(n int, k int) (selectedCol []int) {
	rand.Seed(time.Now().UnixNano())

	tmp := make([]int, n)
	for i := 0; i < n; i++ {
		tmp[i] = i
	}
	for i := 0; i < k; i++ {
		j := i + int(rand.Float64()*float64(n-i))
		tmp[i], tmp[j] = tmp[j], tmp[i]
	}

	selectedCol = tmp[:k]
	sort.Ints(selectedCol)

	return selectedCol
}
