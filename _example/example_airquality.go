package main

import (
	rf ".."
	"log"
	// "github.com/davecgh/go-spew/spew"
)

func main() {
	features, labels := rf.ParseCSV("data/airquality.csv")

	k, estimators, depth := 2, 10, 5
	model := rf.NewForest("regression", k, estimators, depth).Build(features, labels)

	predictions := model.Predict(features)
	log.Println(predictions)

	// Dump the forest.
	// spew.Dump(model)
}
