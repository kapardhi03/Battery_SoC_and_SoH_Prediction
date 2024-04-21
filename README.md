# Battery SoH and SoC Prediction

This project focuses on predicting the State of Health (SoH) and State of Charge (SoC) of a battery using an LSTM (Long Short-Term Memory) neural network model. The model is trained on historical battery data and can make predictions based on user input.

## Features

- Predicts battery SoH and SoC based on various input parameters.
- Utilizes an LSTM model for accurate predictions.
- Provides a user-friendly web interface using Streamlit.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/battery-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd battery-prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure that you have the trained model file (`model.pth`) in the project directory. If not, you can train the model by running the `backend.py` script.

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. The app will open in your default web browser. Enter the required input parameters in the provided fields and click the "Predict" button to get the predicted SoH and SoC values.

## Model Training

The LSTM model is trained using historical battery data stored in an Excel file. The `backend.py` script handles the data preprocessing, model training, and saving of the trained model.

To train the model:

1. Update the `data_path` variable in `backend.py` to point to the correct location of your consolidated data Excel file.

2. Run the `backend.py` script:

    ```bash
    python backend.py
    ```

The script will load the data, preprocess it, train the LSTM model, and save the trained model as `model.pth` in the project directory.

## File Structure

- `app.py`: Streamlit app for the web interface.
- `backend.py`: Script for data preprocessing, model training, and prediction.
- `model.pth`: Trained LSTM model (generated after running `backend.py`).
- `requirements.txt`: List of required dependencies.

## Dependencies

- Streamlit
- Pandas
- NumPy
- PyTorch
- scikit-learn

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
