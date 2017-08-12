package randomforest

import (
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

func ParseCSV(filepath string) (features [][]float64, labels []float64) {
	f, err := os.Open(filepath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	content, _ := ioutil.ReadAll(f)
	lines := strings.Split(string(content), "\n")

	for _, line := range lines {
		line = strings.TrimRight(line, "\r\n")

		if len(line) == 0 {
			continue
		}

		row := strings.Split(line, ",")

		feature := []float64{}
		for _, x := range row[:len(row)-1] {
			f_x, _ := strconv.ParseFloat(x, 64)
			feature = append(feature, f_x)
		}
		features = append(features, feature)

		label, _ := strconv.ParseFloat(row[len(row)-1], 64)
		labels = append(labels, label)
	}

	return features, labels
}
