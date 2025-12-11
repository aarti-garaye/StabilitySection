# Stability Selection
Stability Selection is a powerful technique for feature selection in machine learning and statistics. It combines subsampling with selection algorithms to identify stable features that are consistently selected across different subsets of the data. 

# Overview
This project reproduces and extends the results from Meinshausen & Bühlmann (2010) on Stability Selection, with a focus on:
- Understanding why the Lasso becomes unstable under strong correlations
- Demonstrating how stability selection improves variable selection robustness
- Implementing randomized Lasso + stability selection
- Reproducing Figure 4 from the original paper
- Extending the simulation to more realistic, heterogeneous covariance structures
- Studying violations of the Irrepresentable Condition under standardization

The project Deliverables include:
- A Jupyter notebook that reproduces the key results and figures from the paper
- An extended simulation study with heterogeneous covariance structures
- A detailed report discussing the findings and implications of the results
- Slides summarizing the methodology and key insights from the project
- Well-documented code for all simulations and analyses
- Reproducible Code + Data

StabilitySelection/
│
├── Reports/
│   ├── FinalReport.ipynb          # The main analysis + report notebook
│   ├── FinalReport.html          # Exported report
│
├── Presentation/
│   ├── slides.pdf      # Presentation PDF
|   ├── FinalSlides.pptx # Presentation in PowerPoint
│   └── slides.md # Presentation in markdown
│   └── ucsb-theme.css # Theme for presentation
│
├── Scripts/
│   ├── Code/
|   |   ├── Reproduction.py
|   |   ├── Extension1.py
|   |   ├── Extension2.py
|   |   ├── Extension3.py
│   ├── Raw/
|   |   ├── Darft.ipynb
|   |   ├── Darft.html
│
├── Images/
|   ├── Extension1Basic.png
|   ├── IncreasedVar.png
|   ├── RandomRho.png
|   ├── ReproduceingFig4.png
│
└── README.md                # You are here

# Reproducing Figure 4 (Meinshausen & Bühlmann, 2010)

The code in Scripts/fig4_simulation.py implements:

- The correlated Gaussian design
- True coefficient structure
- Standard Lasso
- Moderate randomized Lasso (α = 0.5)
- Strong randomized Lasso (α = 0.2)
- Half-subsampling stability selection

It produces the full stability-path figure comparing the methods.