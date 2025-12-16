# WGU Assignment – Bureau of Transportation Analysis

**Author:** Raghe Mahamud  
**Date:** 09/04/2025  
**Course:** D602 – Data Analytics (WGU)

---

## Import and Format Script:
The initial step involved reviewing the dataset documentation to identify the relevant columns required for analysis. Due to the large size of the dataset obtained from the Bureau of Transportation Statistics, locating the necessary airport data presented an initial challenge.

After identifying the required fields, the dataset was reduced to include only those columns. Column names were standardized to ensure consistency throughout the project. The data-cleaning script was uploaded to GitLab with multiple commits to demonstrate code progression and refinement.

To support reproducibility and data version control, a DVC command was executed to generate a metadata file for the dataset, which was also committed to GitLab.

---

## Filtering Data:
The dataset was filtered to include only records associated with **DFW Airport**, significantly reducing the dataset size and improving usability.

Additional data-cleaning steps included:
- Removing rows containing missing values
- Stripping leading and trailing whitespace from string fields

The finalized, cleaned dataset was then exported and saved for use in subsequent modeling steps.

---

## MLflow Experiment:
The MLflow experiment was completed by adapting and validating provided code within a Jupyter Notebook environment. The primary change involved updating the input filename to reference the cleaned dataset.

Because column names had been standardized to lowercase, the code was modified accordingly (e.g., replacing references to `YEAR` and `MONTH` with `year` and `month`). After these adjustments, the script executed successfully.

The remaining tasks were completed by running the model, calculating the mean squared error (MSE) and average delay time, and generating the required plots.

---

## MLProject Linking File:
Developing the ML pipeline was the most challenging component of this project. Multiple issues were encountered related to YAML configuration, MLflow integration, and dependency management. These were resolved through iterative troubleshooting using the terminal, including modifying Python scripts, addressing MLflow conflicts, and resolving library dependencies such as `openpyxl`.

Jupyter Notebook files were converted into Python scripts to ensure compatibility with the ML pipeline. Additional fixes addressed JSON handling, OS-related errors, and proper saving of the trained model as `model.pkl`.

Once resolved:
- `D602Task2.py` generated the cleaned dataset
- `PolyRegressor.py` trained the model and saved artifacts
- `log_results.py` logged metrics and outputs to MLflow successfully

The CI/CD pipeline executed without errors, and the full workflow ran successfully via the terminal.

---

## Sources
No external sources were used beyond official **Western Governors University course materials**.
