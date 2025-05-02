import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import time
import random
import sys
import os
from datetime import datetime

# Add 'bolt-job' folder to system path
sys.path.append('bolt-job')
from run import Model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fake loading bar function
def fake_loading(task_name, min_seconds=2, max_seconds=5):
    total_seconds = random.uniform(min_seconds, max_seconds)
    start_time = time.time()

    print(f"\n{task_name}...")
    with tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', dynamic_ncols=True) as pbar:
        for _ in range(100):
            time.sleep(total_seconds / 100)
            elapsed = int(time.time() - start_time)
            pbar.set_description(f"Elapsed Time: {elapsed} sec")
            pbar.update(1)

    print(f"\n{task_name} complete. Total time: {int(time.time() - start_time)} seconds.")

# Clear screen function
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Load the trained model
def load_model():
    fake_loading("Loading Model", min_seconds=2, max_seconds=4)
    model = Model().to(device)
    model.load_state_dict(torch.load('bolt-job/model_final.pt', map_location=device))
    model.eval()
    print("\nModel loaded and ready for evaluation.")
    return model

# Load test data
def load_test_data():
    fake_loading("Loading Testing Data", min_seconds=85, max_seconds=95)
    try:
        df = pd.read_csv("Cleaned_Dataset.csv")
    except FileNotFoundError:
        print("\nCould not find 'Cleaned_Dataset.csv'. Make sure it is in the same folder.")
        return None, None

    test_size = int(0.2 * len(df))
    test_df = df[-test_size:].reset_index(drop=True)

    if 'Status' not in test_df.columns:
        print("\n'Status' column not found in dataset.")
        return None, None

    X_test = test_df.drop(columns=["Status"])
    y_test = test_df["Status"]

    X_test = X_test.select_dtypes(include=[np.number])

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    return X_test_tensor, y_test_tensor

# Train the model from training data
def train_model():
    print("\nTraining starting. Please wait...\n")

    dataset_path = "Cleaned_Dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset '{dataset_path}' not found. Please load or generate it first.")
        return

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    test_size = int(0.2 * len(df))
    train_df = df[:-test_size].reset_index(drop=True)

    X_train = train_df.drop(columns=["Status"])
    y_train = train_df["Status"]

    X_train = X_train.select_dtypes(include=[np.number])
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)

    model = Model().to(device)
    model_path = "bolt-job/model_final.pt"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Existing model loaded for continued training.")
    else:
        print("No existing model found. Starting new training.")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    while True:
        try:
            max_epochs = input("Enter the number of epochs to train (default = 30): ").strip()
            if max_epochs == "":
                max_epochs = 30
            else:
                max_epochs = int(max_epochs)
            if max_epochs <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    model.train()
    epoch_counter = 0
    start_time = datetime.now()

    progress_bar = tqdm(range(max_epochs), desc="Training Model", ncols=100, dynamic_ncols=True, leave=True)

    for epoch in progress_bar:
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()

        epoch_counter += 1
        progress_bar.set_description(f"Epoch {epoch+1}/{max_epochs} | Loss: {loss.item():.4f}")

    duration = (datetime.now() - start_time).total_seconds()

    torch.save(model.state_dict(), model_path)
    print(f"\nModel training complete.")
    print(f"Total additional epochs completed: {epoch_counter}")
    print(f"Training duration: {int(duration)} seconds.")
    print(f"Model saved to '{model_path}'.")

    log_path = "bolt-job/training_log.txt"
    with open(log_path, "a") as log_file:
        log_file.write(f"[{datetime.now()}] Trained {epoch_counter} epoch(s) in {int(duration)} sec\n")

# Evaluate model and print classification report
def evaluate_model(model, X_test, y_test):
    fake_loading("Evaluating Model", min_seconds=2, max_seconds=4)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        outputs = torch.sigmoid(model(X_test))
        preds = (outputs > 0.5).long().squeeze(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print("\nSample Predictions:")
    for i in range(5):
        print(f"Prediction: {all_preds[i]} \tActual: {all_labels[i]}")

# Show the text-based menu
def show_menu():
    print("\n===== MENU =====")
    print("1. Load Model")
    print("2. Load Testing Data")
    print("3. Train Model")
    print("4. Evaluate Model")
    print("5. Quit")

# Main program
def main():
    model = None
    X_test = None
    y_test = None

    while True:
        show_menu()
        choice = input("\nEnter your choice: ")

        if choice == "1":
            clear_screen()
            model = load_model()
        elif choice == "2":
            clear_screen()
            X_test, y_test = load_test_data()
        elif choice == "3":
            clear_screen()
            train_model()
        elif choice == "4":
            if model is None:
                print("\nModel not loaded. Please load the model first.")
            elif X_test is None or y_test is None:
                print("\nTesting data not loaded. Please load testing data first.")
            else:
                evaluate_model(model, X_test, y_test)
        elif choice == "5":
            print("\nExiting program. Goodbye.")
            break
        else:
            print("\nInvalid choice. Please select from the menu.")

if __name__ == "__main__":
    main()

