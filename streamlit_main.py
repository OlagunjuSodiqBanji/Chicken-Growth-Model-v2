import pandas as pd
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import streamlit as st
modelbw8 = joblib.load('random_forest_model_bw8.pkl')
modelbw12 = joblib.load('random_forest_model_bw12.pkl')
modelbw16 = joblib.load('random_forest_model_bw16.pkl')

if "count"not in st.session_state:
    st.session_state.count = 0 

def gompertz_growth_model(t, A, k, tip):
    """
    Calculate the predicted body weight (BW) at age t using the Gompertz growth model.

    Parameters:
    - t (float): Age in days (or any consistent time unit).
    - A (float): Asymptotic adult body weight.
    - k (float): Growth rate constant.
    - tip (float): Age at the inflection point of the growth curve.

    Returns:
    - float: Predicted body weight at age t.
    """
    BW = A * np.exp(-np.exp(-k * (t - tip)))
    return BW


# print(predicted_valuebw8,predicted_valuebw12,predicted_valuebw16)

# print("Gompertz model=",bw8,bw12,bw16)




st.set_page_config(
    page_title="chicken Cross Breeding Model",
    page_icon="🐔",
    layout= "wide"
)
st.title("Chicken Cross breeding growth model")

@st.fragment(run_every=2)
def infomation():
    if st.session_state.count<10:
        st.info(icon="📈",body="The Model Was Trained to fit MN(Pure Breed's) , MA(Pure Breed's), MN X MA, MA X MN ")


infomation()

sh= st.selectbox(
    label="Which Breed Type",
    options=["MN(PureBreed)","MN X MA","MA X MN","MA(PureBreed)",""],
    key="Selectbox"

)
if st.session_state.Selectbox=="Other":
    st.warning(icon="📉",body="It can Accomodate Other Breeds But there wasn't any data to train with other breeds so accuracy of predictions with others is significantly lower than making predictions with training data")
    st.text_input(label="Rooster Breed",key="RoosterBreedType")
    st.number_input(label="Expected weight in grams of adult maturity in adult Rooster Breed",key="RoosterA")

    st.text_input(label="Hen Breed",key="HenBreedType")
    st.number_input(label="Expected weight in grams of adult maturity in adult Rooster Breed",key="HenA")


sh1=st.number_input(
    label= "Weight Of Chicken Hatched In Grams",
    key="eggweight",
    min_value=30,
    max_value=454
)
feed1=st.number_input(
    label= "Weight Of Chicken feed In Grams for 4 weeks",
    key="feed1",
    min_value=20,
    max_value=20
)
feed2=st.number_input(
    label= "Weight Of Chicken feed In Grams for 8 weeks perday ",
    key="feed2",
    min_value=40,
    max_value=40
)

feed3=st.number_input(
    label= "Weight Of Chicken feed In Grams for 12 weeks per day",
    key="feed3",
    min_value=80,
    max_value=80
)

feed4=st.number_input(
    label= "Weight Of Chicken feed In Grams for 16 weeks per day",
    key="feed4",
    min_value=100,
    max_value=100
)

# Define the column names based on your training data
column_names = [
    'Hatch Weight (g)', 'tip', 'k', 'predicted_bw8', 'predicted_bw12',  'predicted_bw16', 'Parent Breed Sire_MN', 'Parent Breed Dam_MN', 'Cross Type_Purebred'
]

# Create a DataFrame with the data and column names
bw8 = gompertz_growth_model(t=32,A=1000,k=0.004,tip=25)
bw12 = gompertz_growth_model(t=48,A=1000,k=0.023,tip=15)
bw16 = gompertz_growth_model(t=64,A=1000,k=0.05,tip=7)
if st.session_state.Selectbox!="Other":

    data_df_bw8 = pd.DataFrame([[st.session_state.eggweight, 25,0.004, bw8, bw12, bw16, True, True, False]], columns=column_names) if st.session_state.Selectbox=="MN(PureBreed)" else pd.DataFrame([[st.session_state.eggweight, 25,0.004, bw8, bw12, bw16, False, False, False]], columns=column_names) if  st.session_state.Selectbox=="MA(PureBreed)" else pd.DataFrame([[st.session_state.eggweight, 25,0.004, bw8, bw12, bw16, True, False, True]], columns=column_names) if st.session_state.Selectbox =="MN X MA" else pd.DataFrame([[st.session_state.eggweight, 25,0.004, bw8, bw12, bw16, False, True, True]], columns=column_names)
    
    data_df_bw12 = pd.DataFrame(
        [[st.session_state.eggweight, 15, 0.023, bw8, bw12, bw16, True, True, False]], columns=column_names
    ) if st.session_state.Selectbox == "MN(PureBreed)" else pd.DataFrame(
        [[st.session_state.eggweight, 15, 0.023, bw8, bw12, bw16, False, False, False]], columns=column_names
    ) if st.session_state.Selectbox == "MA(PureBreed)" else pd.DataFrame(
        [[st.session_state.eggweight, 15, 0.023, bw8, bw12, bw16, True, False, True]], columns=column_names
    ) if st.session_state.Selectbox == "MN X MA" else pd.DataFrame(
        [[st.session_state.eggweight, 15, 0.023, bw8, bw12, bw16, False, True, True]], columns=column_names
    )

    
    data_df_bw16 = pd.DataFrame(
        [[st.session_state.eggweight, 7, 0.05, bw8, bw12, bw16, True, True, False]], columns=column_names
    ) if st.session_state.Selectbox == "MN(PureBreed)" else pd.DataFrame(
        [[st.session_state.eggweight, 7, 0.05, bw8, bw12, bw16, False, False, False]], columns=column_names
    ) if st.session_state.Selectbox == "MA(PureBreed)" else pd.DataFrame(
        [[st.session_state.eggweight, 7, 0.05, bw8, bw12, bw16, True, False, True]], columns=column_names
    ) if st.session_state.Selectbox == "MN X MA" else pd.DataFrame(
        [[st.session_state.eggweight, 7, 0.05, bw8, bw12, bw16, False, True, True]], columns=column_names
    )

else:
    # Write logic to calculate k, tip, and enter digits for the cross breeds and purebreeds
    data_df_bw8 = pd.DataFrame([[st.session_state.eggweight, 25,0.004, bw8, bw12, bw16, True, False, False]], columns=column_names)
    data_df_bw12 = pd.DataFrame([[st.session_state.eggweight, 15, 0.023, bw8, bw12, bw16, True, False, False]], columns=column_names)
    data_df_bw16 = pd.DataFrame([[st.session_state.eggweight, 7, 0.05, bw8, bw12, bw16, True, False, False]], columns=column_names)



# Make the prediction
predicted_valuebw8 = modelbw8.predict(data_df_bw8)
predicted_valuebw12 = modelbw12.predict(data_df_bw12)
predicted_valuebw16 = modelbw16.predict(data_df_bw16)




prediction_df = pd.DataFrame([[(predicted_valuebw8*(st.session_state.feed2/40)-100),(predicted_valuebw8*(st.session_state.feed2/40)),(predicted_valuebw12*(st.session_state.feed3/80)),(predicted_valuebw16*(st.session_state.feed4/100))]],columns=["Predicted Body Weight at 4 weeks","Predicted Body Weight at 8 weeks","Predicted Body Weight at 12 weeks","Predicted Body Weight at 16 weeks",])
st.write(prediction_df)


@st.fragment(run_every=1)
def count():
    st.session_state.count += 1
  


count()


