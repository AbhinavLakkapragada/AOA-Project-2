# AOA Project 2: Network Flow and NP-Completeness

**Authors:** Abhinav Lakkapragada, Rishav Raju Chintalapati 
**Course:** Analysis of Algorithms  
**Date:** December 6, 2025

## Overview

This project implements and analyzes two computational problems from different complexity classes:

1. **Problem A (Network Flow):** Blood Bank Distribution Network - optimal blood allocation using Edmonds-Karp algorithm
2. **Problem B (NP-Complete):** Museum Artwork Arrangement - greedy graph coloring heuristics (Welsh-Powell and DSatur)

## Project Structure

```
AOA-Project-2/
├── project_code.py          # Main implementation
├── project_report.tex        # LaTeX report source
├── project_report.pdf        # Compiled report (if available)
├── outputs/                  # Generated experimental results
│   ├── blood_sanity.csv
│   ├── blood_timing.csv
│   ├── blood_runtime.png
│   ├── museum_sanity.csv
│   ├── museum_timing.csv
│   ├── museum_runtime.png
│   └── museum_rooms.png
└── README.md                 # This file
```

## Requirements

- Python 3.11+
- matplotlib (for plotting)

## Installation

```bash
# Install required package
pip install matplotlib
```

## Running the Code

```bash
# Run all experiments (generates CSV files and PNG plots)
python project_code.py
```

**Expected output:**
- Console output showing timing results for both problems
- CSV files in `outputs/` folder with experimental data
- PNG plots in `outputs/` folder for use in the report

**Runtime:** Approximately 5-10 seconds for all experiments.

## Experimental Results

The code runs the following experiments:

### Problem A: Blood Distribution Network
- Validates flow constraints (20 trials)
- Measures Edmonds-Karp runtime for networks with 5-25 banks/hospitals
- Generates: `blood_sanity.csv`, `blood_timing.csv`, `blood_runtime.png`

### Problem B: Museum Artwork Arrangement
- Validates greedy coloring constraints (20 trials)
- Measures Welsh-Powell and DSatur runtime for 10-60 artworks
- Generates: `museum_sanity.csv`, `museum_timing.csv`, `museum_runtime.png`, `museum_rooms.png`

## Report

The full technical report (`project_report.pdf`) includes:
- Formal problem abstractions
- Polynomial reductions with correctness proofs
- Algorithm descriptions and complexity analysis
- Experimental validation of theoretical bounds

**To compile the LaTeX report:**
1. Upload `project_report.tex` to [Overleaf](https://www.overleaf.com)
2. Create `outputs/` folder and upload the PNG files
3. Compile using pdfLaTeX

## Key Results

| Problem | Algorithm | Complexity | Optimality |
|---------|-----------|------------|------------|
| Blood Distribution | Edmonds-Karp | O(VE²) | Guaranteed |
| Museum Arrangement | Welsh-Powell | O(n²) | Approximation |
| Museum Arrangement | DSatur | O(n²) | Approximation |

## References

See `project_report.tex` for complete bibliography including:
- Cormen et al. (CLRS)
- Garey & Johnson (NP-Completeness)
- Edmonds & Karp (Network Flow)
- Welsh & Powell, Brélaz (Graph Coloring)
