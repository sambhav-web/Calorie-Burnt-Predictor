import streamlit as st
import pandas as pd
import dill
with open('label_encoder.pkl','rb') as f:
  le=dill.load(f)
with open('one_hot_encoder.pkl','rb') as f:
  ohe=dill.load(f)
with open('standard_scaler.pkl','rb') as f:
  scaler=dill.load(f)
with open('model.pkl','rb') as f:
  model=dill.load(f)
df=pd.read_csv('calories.csv')
st.set_page_config(
  page_title="Burnt Calorie Predictor",
  page_icon='🏋🏻',
  layout='centered'
)
st.title('💪 Calories Burnt Prediction')
st.write('This model is XGBregressor based that predicts the amount of calories burnt in workout')
st.info("Provide the details needed to make the prediction: ")
cols=st.columns(2)
with cols[0]:
  gen=st.selectbox('Select Your Gender: ',df['Gender'].unique())
  age=st.number_input('Enter Your Age: ',20,79)
  time=st.slider('Enter Time Duration :',df['Duration'].min(),df['Duration'].max(),10,1,help='Enter the time duration in minutes')
with cols[1]:
  hr=st.number_input('Select the average Heart Rate',df['Heart_Rate'].min(),df['Heart_Rate'].max())
  temp=st.number_input('Select the average Body Temperature',df['Body_Temp'].min(),df['Body_Temp'].max(),help='Enter value in Celcius')
  bmi_value=st.number_input('Enter Bmi value :',19.227688,29.069767,step=0.000001)
age_bmi=age*bmi_value
if bmi_value<18.5:
  bmi_cat='Underweight'
elif bmi_value>=18.5 and bmi_value<25:
  bmi_cat='Normal'
elif bmi_value>=25 and bmi_value<30:
  bmi_cat='Overweight'
else:
  bmi_cat='Obese'
def calories_burnt(gen, time, hr, temp, age_bmi, bmi):
    input_df = pd.DataFrame({
        'Gender': [gen],
        'Duration': [time],
        'Heart_Rate': [hr],
        'Body_Temp': [temp],
        'Age_BMI_Interaction': [age_bmi],
        'BMI_Category': [bmi]
    })
    input_df['Gender'] = le.transform(input_df['Gender'])
    features_encoded = ohe.transform(input_df[['BMI_Category']])
    features_df = pd.DataFrame(features_encoded, columns=ohe.get_feature_names_out())
    input_df = pd.concat([input_df.drop(columns=['BMI_Category']), features_df], axis=1)
    cols_to_scale = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age_BMI_Interaction']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    return model.predict(input_df)[0]
if st.button('Calorie Burnt',type='primary'):
  container=st.container(border=True)
  calorie=calories_burnt(gen,time,hr,temp,age_bmi,bmi_cat)
  container.subheader(f'✅ Congratulation!,You have successfully burnt {calorie} Calories.')