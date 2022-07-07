import streamlit as st
import pickle
import numpy as np


data = pickle.load(open('dataset.pkl', 'rb'))
model = pickle.load(open('logistic_reg.pkl', 'rb'))


st.title('Medical Sample Collection')
agent_id = st.selectbox('Select Agent ID', data['Agent ID'].unique())
slot = st.selectbox('Select Slot for Booking', ['06:00 to 21:00 (Home)' , '19:00 to 22:00 (working person)', '06:00 to 18:00 (Collect at work place)'])
gender = st.radio('Select Gender', ['Female', 'Male'])
storage = st.selectbox('Storage of Specimen', ['Vacuum blood collection tube', 'Urine culture transport tube', 'Disposable plastic container'])
distance = np.log(st.number_input('Distance Between Patient and Agent in Meters'))
collection_time = st.number_input('Specimen collection Time in minutes')
patient_from = st.number_input('Patient Available From', min_value=1, value=20)
if st.checkbox('Show Instruction 1'):
        st.text('In "Patient Available From" input the time when patient is available for test\n'
                'Eg.: Patient is available from 13(1PM) to 14(2PM)\n'
                'Note: value should be in 24-hour format')
patient_to = st.number_input('Patient Available Upto', min_value=1, value=21)
if st.checkbox('Show Instruction 2'):
        st.text('In "Patient Available Upto" input the time when patient is available upto for test\n'
                'Eg.: Patient is available from 13(1PM) to 14(2PM)\n'
                'Note: value should be in 24-hour format')
agent_before = st.number_input('Agent Arrived Before', min_value=1, value=21)
if st.checkbox('Show Instruction 3'):
        st.text('Eg.: Agent will reach before 14(2PM)')

if st.button('Predict Timing'):

        if slot == '06:00 to 18:00 (Collect at work place)':
            slot = 0
        elif slot == '06:00 to 21:00 (Home)':
            slot = 1
        elif slot == '19:00 to 22:00 (working person)':
            slot = 2

        if gender == 'Female':
            gender = 0
        elif gender == 'Male':
            gender = 1

        if storage == 'Disposable plastic container':
            storage = 0
        elif storage == 'Urine culture transport tube':
            storage = 1
        elif storage == 'Vacuum blood collection tube':
            storage = 2

        query = np.array([agent_id, slot, gender, storage, distance, collection_time, patient_from, patient_to, agent_before])
        query = query.reshape(1, 9)

        result = model.predict(query)

        if result == 24:
            st.success(f'Agent will reach within {24} minutes')
        elif result == 34:
            st.success(f'Agent will reach within {34} minutes')
        elif result == 39:
            st.success(f'Agent will reach within {39} minutes')
        elif result == 49:
            st.success(f'Agent will reach within {49} minutes')
        elif result == 54:
            st.success(f'Agent will reach within {54} minutes')
        else: 
            st.success(f'Agent will reach within {64} minutes')
            st.write('Agent cannot reach you, your location is too far')

