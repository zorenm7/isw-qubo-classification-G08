### Generated Code Analysis

The generated code for the training and prediction module is solid and well-structured. Utilizing `argparse` `subparsers` is the cleanest engineering choice to handle multi-functional command-line scripts.

Here are a couple of technical details worth addressing before finalizing the file:

* **Confusion Matrix Label Order:** The LLM set `labels=[0, 1]`. Looking at the example in the professor's PDF (page 10, lines 360-363), the label array expects the reverse order: `[1, 0]`. University automated grading systems are often rigid regarding JSON parsing. I recommend manually modifying that line in the final code to `confusion_matrix(y_true, y_pred, labels=[1, 0])`.
* **Numpy Handling:** There is a small hack on line 80 (`if 'np' in globals()`) because the LLM forgot to import `numpy` at the top of the file, resorting to a list comprehension as a fallback. You just need to add `import numpy as np` among the dependencies at the top to make the code cleaner and more efficient.

---

We are ready for the final step: validation and the user interface. Structuring automated tests that cover edge cases is a fundamental principle of Software Engineering. For the GUI, we will explicitly request the use of **Streamlit**, as it allows creating reactive web interfaces for data-driven applications with very few lines of code, proving much more modern and cleaner than `tkinter`.

### Prompt 4: Automated Tests (Pytest) and GUI (Streamlit)

**Copy and paste to the LLM:**

> **Role:** Act as a QA Automation Engineer and Frontend Python Developer.
> **Task:** Write the automated test suite and the graphical user interface for our QUBO Feature Selection project.
> 
> **Part 1: Automated Tests (`tests/test_pipeline.py`)**
> **Constraints & Specifications:**
> * Use `pytest`.
> * The tests must assume the existence of a small mock dataset located at `data/sample_test_dataset.csv`.
> * Write specific test functions to verify the following exact requirements:
>   1. Preprocessing produces only numerical columns.
>   2. Preprocessing correctly handles missing/null values (drops columns exceeding the threshold).
>   3. Normalization produces a valid dataset (e.g., checks if values are scaled).
>   4. Feature selection produces a valid binary vector (only 0s and 1s).
>   5. The number of selected features is approximately the requested percentage ($\approx 20\%$).
>   6. Training successfully produces and saves a `.joblib` model to disk.
>   7. Prediction successfully produces a CSV with the required columns (`row_n`, `target`, `prediction`, `score`).
> * Ensure tests create temporary directories/files for outputs to avoid polluting the main project directory, using pytest's `tmp_path` fixture.
> 
> **Part 2: Graphical User Interface (`src/qubo_project/gui.py`)**
> **Constraints & Specifications:**
> * Use `streamlit` to build the GUI.
> * The interface must allow a human user to logically step through the pipeline:
>   1. Select/Upload an input dataset (CSV).
>   2. Execute the preprocessing phase (with a slider or number input for `minPercValid`).
>   3. Execute the QUBO feature selection (with inputs for `percSelected`, `allowance`, `alpha_computations`).
>   4. Execute model training (dropdown to select `random_forest`, `logistic_regression`, or `gradient_boosting`).
>   5. Execute predictions on the test set.
>   6. Visualize the main outputs (e.g., display the classification stats JSON, the ROC-AUC, and the selected features).
> * Include basic error handling (e.g., displaying an error message if the user tries to train a model before running feature selection).
> * Ensure the GUI imports and uses the functions from `preprocessing.py`, `feature_selection.py`, and `model.py` rather than duplicating their logic.