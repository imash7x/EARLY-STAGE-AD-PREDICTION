import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

voting_classifier = joblib.load("voting_classifier.pkl")
stage_classifier = joblib.load("APPLICATION/stage_classifier.pkl")

scaler = MinMaxScaler().fit([[0, 0, 0, 0, 0, 0, 0, 0]])

def preprocess_input(user_input):
    user_input['M/F'] = 1 if user_input['M/F'] == 'M' else 0
    scaled_input = scaler.transform([[user_input['Age'], user_input['EDUC'], user_input['SES'],
                                      user_input['MMSE'], user_input['CDR'], user_input['eTIV'],
                                      user_input['nWBV'], user_input['ASF']]])
    return scaled_input

def predict_stage(user_input):
    try:
        inputdata = [[user_input['eTIV'],user_input['nWBV'], user_input['ASF']]]

        stage_prediction = stage_classifier.predict(inputdata)
        stage_mapping = {
            0: "Non-Dementiated",
            1: "Moderate-Dementiated",
            2: "Severe-Dementiated"
        }
        stage_result = stage_mapping.get(stage_prediction[0], "Unknown")
        return stage_result
    except Exception as e:
        st.error("An error occurred during stage prediction.")

def main():
    st.title("EARLY STAGE PREDICTION OF ALZHEIMER'S DISEASE")
    with st.form("dementia_prediction_form"):
        st.write("Please provide the following information: ")
        st.write("Note: Please fill in the input in the specified format in the text field.")
        
        gender = st.radio("Gender", ("Male", "Female"), index=0)
        
        age = st.number_input("Age", min_value=0, max_value=100, value=None, 
                              placeholder="00", step=1)
        
        edu = st.number_input("Years of Education", min_value=0, max_value=30, value=None, 
                                    placeholder="00", step=1)
        
        ses = st.number_input("Socioeconomic Status", min_value=1, max_value=5, value=None, 
                              placeholder="0", step=1)
        
        mmse = st.number_input("Mini-Mental State Examination (MMSE)", min_value=0, max_value=30, value=None, 
                               placeholder="00", step=1)
        
        cdr = st.number_input("Clinical Dementia Rating (CDR)", min_value=0.0, max_value=2.0, value=None, 
                              placeholder="0.0", step=0.1)
        
        etiv = st.number_input("Estimated Total Intracranial Volume (eTIV)", min_value=0, max_value=2000, value=None, 
                               placeholder="0000", step=1)
        
        wbv = st.number_input("Normalized Whole Brain Volume (nWBV)", min_value=0.0, max_value=1.0, value=None, 
                              placeholder="0.000", step=0.001, format="%.3f")
        
        asf = st.number_input("Atlas Scaling Factor (ASF)", min_value=0.0, max_value=2.0, value=None, 
                              placeholder="0.000", step=0.001, format="%.3f")

        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                if not all([age, edu, ses, mmse, etiv, wbv, asf]):
                    st.error("Please fill in all input fields.")
                else:
                    user_input = {
                        'M/F': 'M' if gender == "Male" else 'F',
                        'Age': age,
                        'EDUC': edu,
                        'SES': ses,
                        'MMSE': mmse,
                        'CDR': cdr,
                        'eTIV': etiv,
                        'nWBV': wbv,
                        'ASF': asf
                    }

                    X_custom_input = preprocess_input(user_input)
                    voting_prediction = voting_classifier.predict(X_custom_input)

                    if voting_prediction == 1:
                        st.write("Prediction: Dementiated")
                        stage_result = predict_stage(user_input)
                        st.write("Stage Prediction:", stage_result)
                    else:
                        st.write("Prediction: Non-Dementiated")
            except Exception as e:
                st.error("An error occurred during prediction.")
                

if __name__ == "__main__":
    main()
