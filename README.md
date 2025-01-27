# On Coverage of Test Suite Accuracy

This repository accompanies the project **"On Coverage of Test Suite Accuracy"**, which provides an in-depth analysis and improvements to the Test Suite Accuracy (TSA) metric. TSA is a method for determining semantic equivalence between SQL queries. This project identifies failure cases in TSA and proposes a robust neighbor query generation approach to mitigate false positives.

---

## Abstract

The Test Suite Accuracy metric was introduced to evaluate the semantic equivalence of SQL queries by generating neighbor queries and comparing query outputs. While TSA resolves all known false negatives, it still produces false positives under specific scenarios. In this project:
- We analyze TSA to uncover seven unique failure cases.
- We propose an enhanced neighbor query generation method using statistical values, which significantly reduces false positives.
- We identify edge cases that require additional exploration, such as queries with string values and wildcard characters.

---

## Methodology

### Analysis of Failure Cases
We identified scenarios where the TSA metric fails, leading to false positives. Key examples include:
1. **Queries with Maximum/Minimum Conditional Values**: Incorrect equivalence due to boundary conditions.
2. **Join and Group By Differences**: Failure to distinguish between groupings on different columns.
3. **Wildcard Characters**: Misidentification of equivalence in string pattern matching queries.
4. **Logical Operators and Inclusivity Conditions**: Incorrect results in queries involving "BETWEEN" or logical conditions.
5. **Join Condition with Column Selection Differences**: Equivalence misidentification when selecting columns from different joined tables.

### Proposed Solution
To improve TSA, we enhance the neighbor query generation by:
- Introducing statistical values (minimum, mean, and maximum) derived from the dataset.
- Generating neighbors by replacing numeric values in queries with these statistical values.
- Reducing the number of false positives for numerical queries without increasing computational complexity.

---

## Dataset

The SPIDER dataset was used for experiments. It includes:
- **10,181** human-labeled complex SQL queries.
- **5,693** questions across multiple databases.

---

## Experiments and Results

### Key Insights
- The enhanced neighbor query generation resolved **all false positive scenarios** for numerical queries.
- Failure cases with string values and wildcard characters remain challenging and are highlighted as future work.

### Algorithm Overview
The proposed neighbor generation algorithm involves:
1. Tokenizing the predicted query to identify numerical columns.
2. Generating statistical queries to retrieve values (min, mean, max).
3. Creating neighbor queries by plugging permutations of statistical and original values into the numerical slots.

---

## How to Execute

To evaluate the improvements in TSA, execute the following:

```bash
python evaluation.py --gold sqlresults/dev_gold_experiment.txt --pred sqlresults/dev_preds_experiment.txt --db database --table tables.json --etype exec --plug_value
```

Download the dataset [here](https://drive.google.com/file/d/11RaGF2u1LtLWqirGTfltiSY_E1D4GHhR/view?usp=sharing).

---

## Results Visualization

- **Neighbor Query Generation**:
  - Enhanced algorithm creates distinct variations, resolving numerical query false positives.
- **Edge Cases**:
  - Visualizations of unresolved string and wildcard failures are documented for reference.

---

## Future Work

1. Incorporate advanced SQL features (e.g., window functions) for TSA evaluation.
2. Address challenges in neighbor generation for string and wildcard values.
3. Optimize neighbor generation by prioritizing impactful permutations.

---

## References

This project builds upon prior work in SQL query evaluation, including:
- Zhong et al. on TSA metrics.
- SPIDER benchmark for Text-to-SQL evaluation.

For detailed results, refer to the [project report](https://drive.google.com/file/d/11RaGF2u1LtLWqirGTfltiSY_E1D4GHhR/view?usp=sharing).

---
