package gorf

import ()

type Tree struct {
	Root          *Node
	maxDepth      int
	CriterionFunc Criterion
}

type Criterion func(labels []float64) float64

func NewTree(task string, depth int) *Tree {
	t := &Tree{}
	t.Root = NewNode()
	t.Root.depth = 0
	t.maxDepth = depth

	switch task {
	case "regression", "mse":
		t.CriterionFunc = mse
	case "classification", "gini":
		t.CriterionFunc = gini
	default:
		t.CriterionFunc = gini
	}
	return t
}

// Build a tree.
func (t *Tree) Build(features [][]float64, labels []float64) {
	t.Root.Add(features, labels, t.CriterionFunc)
}

// Prediction by base learner.
func (t *Tree) Predict(feature []float64) float64 {
	return t.Root.Predict(feature)
}

// TODO: Imple prune tree
func (t *Tree) Prune() {}
