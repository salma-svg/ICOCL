import streamlit as st
from Optimisation import optimiser_nuance 
import os

def main():
    st.title("Iron Casting Optimization")

    # Input field for directory
    directory_path = st.text_input("Enter the directory path", "data")

    if st.button("Process Directory"):
        if os.path.isdir(directory_path):
            st.write("Directory found! Processing files...")
        else:
            st.error("Required files not found in the directory.")
    # List files in the directory
    files = os.listdir(directory_path)
    st.write("Files in directory:", files)
    
    # Load necessary files
    data_file = next((file for file in files if file.endswith("data_nettoyées.xlsx")), None)
    mp_file = next((file for file in files if file.endswith("MP.xlsx")), None)
    constraints_file = next((file for file in files if file.endswith("Contraintes_composants.CSV")), None)
    
    if data_file and mp_file and constraints_file:
        st.write(f"Loading files: {data_file}, {mp_file}, {constraints_file}")
        
        nuance = st.selectbox("Select the nuance", ["GS 400-15", "GS 450-10", "GS 500-7", "GS 600-3"])
        faire_prediction = st.selectbox("Faire prediction (oui/non)", ["oui", "non"])
            
    else:
        st.write("The provided path is not a valid directory")

    if st.button("Optimize"):
        st.write("Optimize button clicked")

        st.write(f"Nuance: {nuance}")
        st.write(f"Directory Path: {directory_path}")
        st.write(f"Faire Prediction: {faire_prediction}")

        with st.spinner("Running optimization..."):
            try:
                result = optimiser_nuance(nuance, directory_path, faire_prediction == "oui")
                st.success("Optimization complete!")
                st.write("Composition en matières premières :\n")
                st.write(result[0])
                st.write("Composition chimiques :\n")
                st.write(result[1])
                st.write("Prédictions :\n")
                st.write(result[2])

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
