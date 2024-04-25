import streamlit as st
import pandas as pd
import torch
from model import predict_soh_soc, fit_scalers, model


train_data = pd.read_excel('/Users/kapardhikannekanti/Battery_SoC_and_SoH_Prediction/consolidated data.xlsx')
X_train = train_data[['speed', 'distance', 'remainingrange', 'batteryvoltage', 'batterycurrent',
                      'cellmaxvoltage', 'cellminvoltage', 'mcu_dcvoltage', 'mcu_dccurrent',
                      'mcu_acrmscurrent', 'mcu_speed', 'mcu_temperature']].values
y_train_soh = train_data['batterysoh'].values.reshape(-1, 1)
y_train_soc = train_data['batterysoc'].values.reshape(-1, 1)



scaler_X, scaler_y_soh, scaler_y_soc = fit_scalers(X_train, y_train_soh, y_train_soc)



def main():
    st.title("Battery SoH and SoC Prediction")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df)

    speed = st.number_input("Enter speed:", value=0.0)
    distance = st.number_input("Enter distance:", value=0.0)
    remaining_range = st.number_input("Enter remaining range:", value=0.0)
    battery_voltage = st.number_input("Enter battery voltage:", value=0.0)
    battery_current = st.number_input("Enter battery current:", value=0.0)
    cell_max_voltage = st.number_input("Enter cell max voltage:", value=0.0)
    cell_min_voltage = st.number_input("Enter cell min voltage:", value=0.0)
    mcu_dc_voltage = st.number_input("Enter MCU DC voltage:", value=0.0)
    mcu_dc_current = st.number_input("Enter MCU DC current:", value=0.0)
    mcu_ac_rms_current = st.number_input("Enter MCU AC rms current:", value=0.0)
    mcu_speed = st.number_input("Enter MCU speed:", value=0.0)
    mcu_temperature = st.number_input("Enter MCU temperature:", value=0.0)
    max_temperature = st.number_input("Enter Max TemperratureL:" , value=0.0)
    battery_changing_status = st.number_input("Enter status:",value=False)
    input_data = pd.DataFrame([[speed, distance, remaining_range, battery_voltage, battery_current,
                                cell_max_voltage, cell_min_voltage, mcu_dc_voltage, mcu_dc_current,
                                mcu_ac_rms_current, mcu_speed, mcu_temperature]],
                              columns=['speed', 'distance', 'remainingrange', 'batteryvoltage', 'batterycurrent',
                                       'cellmaxvoltage', 'cellminvoltage', 'mcu_dcvoltage', 'mcu_dccurrent',
                                       'mcu_acrmscurrent', 'mcu_speed', 'mcu_temperature'])

    if st.button("Predict"):
        soh, soc = predict_soh_soc(model, input_data, scaler_X, scaler_y_soh, scaler_y_soc)
        st.success(f"Predicted SOH: {soh:.2f}")
        st.success(f"Predicted SOC: {soc:.2f}")

if __name__ == '__main__':
    main()