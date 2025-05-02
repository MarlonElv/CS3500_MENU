# Documentation for Neural Network Training Menu Program

---

## Line-by-Line Overview

### Lines 3–11  
Imports necessary Python libraries:
- `torch`: Deep learning framework
- `pandas`, `numpy`: Data loading and processing
- `tqdm`: Progress bar utilities
- `sys`, `os`: File and system operations

### Lines 14–15  
Adds the `'bolt-job'` folder to the system path to enable importing the custom `Model` class from `run.py`.

### Lines 16–18  
Sets the computation device to CUDA (GPU) if available, otherwise defaults to CPU.

### Lines 21–31  
Defines `fake_loading()` to simulate a loading bar using `tqdm`. This gives feedback to the user during time-consuming operations like loading or processing.

### Lines 34–35  
Defines `clear_screen()` to clear the terminal window for improved readability based on the operating system.

### Lines 38–44  
`load_model()` loads the trained model from disk (`model_final.pt`), places it in evaluation mode, and returns the model for inference or evaluation.

### Lines 47–71  
`load_test_data()` handles reading `Cleaned_Dataset.csv`, simulates a long loading time, and converts test features and labels into PyTorch tensors.

### Lines 74–140  
`train_model()` is the main training function:
- Loads or checks for an existing cleaned dataset
- Loads the model if saved, or initializes a new one
- Prompts the user for number of epochs to train
- Runs the training loop and updates weights
- Saves the model to disk
- Appends training progress and duration to a log file (`training_log.txt`)

### Lines 143–160  
`evaluate_model()` runs the model on test data and displays:
- Classification report (precision, recall, F1)
- Accuracy
- 5 sample predictions

### Lines 163–167  
`show_menu()` displays a simple numbered menu for user selection.

### Lines 170–194  
The `main()` function loops indefinitely to:
- Let the user load the model or test data
- Train the model (new or continued)
- Evaluate the model
- Exit the application

### Line 196  
Ensures the program runs only when executed directly (not when imported).

---

## Overall Summary

This command-line neural network trainer enables:

- Continuous model training via saved weights
- Custom user-defined epoch control
- Evaluation using live metrics (classification report + accuracy)
- Fake loading bars for UX feedback
- Logging of each training session in `training_log.txt`

It is built using PyTorch and is modular enough to expand further (e.g., add new menu options, alternate models, or UI enhancements).

