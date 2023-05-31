import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('mlp_model.pkl', 'rb'))

tab_model, tab_figures = st.tabs(["Model", "Figures"])

with tab_model:
    html_temp = """
                <div style="background-color:teal ;padding:0px">
                <h2 style="color:white;text-align:center;">Heart Disease Predictor</h2>
                <h5 style="color:white;text-align:center;">An ML Classification Model</h5>
                </div>
                """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.divider()

    st.markdown("<h5 style='text-align: center; color: teal;'>General Health Attributes</h5>", unsafe_allow_html=True)
    col11, col12 = st.columns(2)
    with col11:
        age = st.number_input('Age', min_value=0, max_value=125, value=25, step=1)
        
        #db_select = ['0: No Diabetic History', '1: Diabetic Hisotry']
        #db = int(st.selectbox('Diabetes History', db_select)[0])
    with col12:
        sex = 1 if st.selectbox('Biological Sex', ['Male', 'Female']) == 'Male' else 0
        
        #smoke_select = ['0: Non-Smoker', '1: Smoker']
        #smoke = int(st.selectbox('Smoking History', smoke_select)[0])
        
    st.divider()
    st.markdown("<h5 style='text-align: center; color: teal;'>Pain Related Attributes</h5>", unsafe_allow_html=True)

    col21, col22 = st.columns(2)
    with col21:
        cp_select = ["1: Typical Angina", "2: Atypical Angina", "3: Non-Anginal Pain", "4: Asymptomatic"]
        cp = int(st.selectbox("Chest Pain (Angina)", cp_select)[0])

    with col22:
        exang_select = ["0: Not Present", "1: Present"]
        exang = int(st.selectbox('Exercise Induced Angina', exang_select)[0])

    st.divider()
    st.markdown("<h5 style='text-align: center; color: teal;'>Blood Related Attributes</h5>", unsafe_allow_html=True)

    col31, col32, col33 = st.columns(3)
    with col31:
        restbps = st.number_input('Resting Bloodpressure (mmHg)', min_value=40, max_value=200, value=120, step=1)
    with col32:
        chol = st.number_input('Cholesterol (mg/dL)', min_value=25, max_value=300, value=175, step=1)
    with col33:
        fbs_select = ['0: FBS < 120 mg/dL', '1: FBS > 120 mg/dL']
        fbs = int(st.selectbox('Fasting Blood Sugar', fbs_select)[0])
        
    st.divider()
    st.markdown("<h5 style='text-align: center; color: teal;'>Other Attributes</h5>", unsafe_allow_html=True)

    col41, col42, col43 = st.columns(3)
    with col41:    
        oldpeak = st.number_input('ST-segment Depression', min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        
        slope_select = ["1: Upsloping", "2: Flat", "3: Downsloping"]
        slope = int(st.selectbox('ST-segment Slope', slope_select)[0])
                
    with col42:
        thalach = st.number_input('Maximum Heartrate (bpm)', min_value=30, max_value=230, value=180, step=1)
        
        restecg_select = ["0: Normal", "1: ST-T Abnormalities", "2: Probable Hypertrophy"]
        restecg = int(st.selectbox('Resting Electrocardiograph', restecg_select)[0])  
        
    with col43:
        ca = st.number_input('Coronary Arteries by Fluoroscopy', min_value=0, max_value=3, step=1)
        
        thal_select = ["3: Normal", "6: Fixed Defect", "7: Reversable Defect"]
        thal = int(st.selectbox('Thalassemia', thal_select)[0])

    st.divider()

    result = '0'
    result = model.predict([[age, sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0]

    st.info(f"Heart Disease Risk: {result}")    
    st.divider()
    
with tab_figures:
    cm_standard = pickle.load(open('cm_standard.pkl', 'rb'))
    df_standard = pd.DataFrame(cm_standard).transpose()
    cm_less = pickle.load(open('cm_less.pkl', 'rb'))
    df_less = pd.DataFrame(cm_less).transpose()
    cm_more = pickle.load(open('cm_more.pkl', 'rb'))
    df_more = pd.DataFrame(cm_more).transpose()
    
    st.caption("Figure 1: Confusion Matrix for Standard CNN")
    st.image('figures/cm_standard.png')
    st.caption("Table 1: Classification Report for Standard CNN")
    st.dataframe(df_standard, width=683)
    
    st.caption("Figure 2: Confusion Matrix for Less Attributes CNN")
    st.image('figures/cm_less.png')
    st.caption("Table 2: Classification Report for Less Attributes CNN")
    st.dataframe(df_less, width=683)
    
    st.caption("Figure 3: Confusion Matrix for More Attributes CNN")
    st.image('figures/cm_more.png')
    st.caption("Table 3: Classification Report for More Attributes CNN")
    st.dataframe(df_more, width=683)
