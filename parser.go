package gorf

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
	rows := strings.Split(string(content), "\n")

	for i, row := range rows {
		row = strings.TrimRight(row, "\r\n")

		// remove header
		if len(row) == 0 || i == 0 {
			continue
		}

		cols := strings.Split(row, ",")

		feature := []float64{}
		for _, x := range cols[:len(cols)-1] {
			f_x, _ := strconv.ParseFloat(x, 64)
			feature = append(feature, f_x)
		}
		features = append(features, feature)

		label, _ := strconv.ParseFloat(cols[len(cols)-1], 64)
		labels = append(labels, label)
	}

	return features, labels
}
