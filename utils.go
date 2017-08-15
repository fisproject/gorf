package gorf

import ()

func unique(labels []float64) []float64 {
	uniq := make([]float64, 0, len(labels))
	encountered := map[float64]bool{}
	for _, v := range labels {
		if !encountered[v] {
			encountered[v] = true
			uniq = append(uniq, v)
		}
	}
	return uniq
}

func count(labels []float64, elem float64) (cnt int) {
	for _, v := range labels {
		if v == elem {
			cnt++
		}
	}
	return cnt
}

func selectCol(features [][]float64, index int) (col []float64) {
	for i := 0; i < len(features); i++ {
		col = append(col, features[i][index])
	}
	return col
}

func divide(features [][]float64, labels []float64, index int, threshold float64) (lhs, rhs [][]float64, lhsl, rhsl []float64) {
	for i := 0; i < len(features); i++ {
		if features[i][index] <= threshold {
			lhs = append(lhs, features[i])
			lhsl = append(lhsl, labels[i])
		} else {
			rhs = append(rhs, features[i])
			rhsl = append(rhsl, labels[i])
		}
	}
	return lhs, rhs, lhsl, rhsl
}

func getThresholds(sorted []float64) (thresholds []float64) {
	if len(sorted) == 1 {
		return sorted
	}

	lhs := sorted[:len(sorted)-1]
	rhs := sorted[1:]

	for i := 0; i < len(sorted)-1; i++ {
		thresholds = append(thresholds, (lhs[i]+rhs[i])/2.0)
	}
	return thresholds
}
