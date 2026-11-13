Data Wrangling Practice

## Overview
Complete data wrangling pipeline on automobile dataset: missing value handling, data formatting, normalization, binning, indicator variables, and correlation analysis.

## Original Notebooks (17)
Combined from 17 individual data wrangling lab notebooks.

## Key Skills Demonstrated
- Missing value detection and treatment (mean, mode, drop)
- Data type conversion and validation
- Unit conversion (mpg → L/100km, inches → cm, lbs → kg)
- Min-Max normalization and Z-score standardization
- Binning continuous variables (equal-width, quantile)
- One-hot encoding (indicator/dummy variables)
- Correlation analysis and heatmap visualization

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy requests
python main.py
```

## Outputs
- `normalization_comparison.png` - Original vs normalized distributions
- `correlation_heatmap.png` - Full correlation matrix

## Key Findings
- Engine size, horsepower, curb weight most correlate with price
- Normalization essential for distance-based algorithms
- Binning reveals non-linear relationships
- 17% of original data had missing values ("?")
