package randomforest

import ()

type Tree struct {
	Root          *Node
	maxDepth      int
	CriterionFunc Criterion
}

type Criterion func(labels []float64) float64

func NewTree(f Criterion, depth int) *Tree {
	t := &Tree{}
	t.Root = NewNode()
	t.Root.depth = 0
	t.maxDepth = depth
	t.CriterionFunc = f
	return t
}

func (t *Tree) Build(features [][]float64, labels []float64) {
	t.Root.Grow(features, labels, t.CriterionFunc)
}

func (t *Tree) Predict(feature []float64) float64 {
	return t.Root.Predict(feature)
}

// TODO: imple prune tree
func (t *Tree) Prune() {}
