gorf - Random Forest implemtation in Go
====

## Overview
Simple Random Forest implemtation in Go.

## Usage

```go
package main

import (
	rf ".."
	"log"
	// "github.com/davecgh/go-spew/spew"
)

func main() {
	features, labels := rf.ParseCSV("data/iris.csv")

	k, estimators, depth := 2, 10, 5
	model := rf.NewForest("classification", k, estimators, depth).Build(features, labels)

	predictions := model.Predict(features)
	log.Println(predictions)

	// Dump the forest.
	// spew.Dump(model)
}
```

## Licence
[MIT](http://opensource.org/licenses/MIT)

## Author
[t2sy](https://github.com/fisproject)
