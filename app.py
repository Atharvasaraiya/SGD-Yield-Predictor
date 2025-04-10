# Import Important Library
import joblib
import streamlit as st 
from PIL import Image
import pandas as pd

# Load Model, Scaler, Polynomial Features, and Training Feature Columns
model = joblib.load('model.pkl')
sc = joblib.load('sc.pkl')
pf = joblib.load('pf.pkl')
feature_columns = joblib.load('features.pkl')  # You need to save this during model training

# Load dataset for dropdown options
df_main = pd.read_csv('main.csv')
df_final = pd.read_csv('test.csv')  # used only for dummy structure

# Load Image
image = Image.open('img.png')

# Streamlit Function
def main():
    st.image("use.jpg", use_column_width=True)

    st.markdown("<h2 style='text-align:center; color:#2e7d32;'>🌾 Yield Crop Prediction</h2>", unsafe_allow_html=True)

    html_temp = '''
    <div style='background-color:#2e7d32; padding:1.5vw; border-radius:0.8vw; margin-bottom: 1.5vw'>
        <h3 style='color:#ffffff; text-align:center;'>Yield Crop Prediction Machine Learning Model</h3>
    </div>
    <h4 style='color:#2e7d32; text-align:center;'>🌱 Please Enter Input</h4>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("🌍 Country", df_main['area'].unique()) 
        average_rainfall = st.number_input('🌧️ Average Rainfall (mm/year)', value=0.0)

    with col2:
        crop = st.selectbox("🌾 Crop", df_main['item'].unique()) 
        presticides = st.number_input('🧪 Pesticides Use (tonnes)', value=0.0)

    avg_temp = st.number_input('🌡️ Average Temperature (°C)', value=0.0)

    input_data = [country, crop, average_rainfall, presticides, avg_temp]

    if st.button('🚀 Predict'):
        result = prediction(input_data)
        if result:
            st.markdown(f"""
            <div style='background-color:#66bb6a; padding:1.5vw; border-radius:0.8vw; margin-top:2vw'>
                <h4 style='color:#003300; text-align:center;'>Prediction Result: {result}</h4>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff;
                color: #2e7d32;
                font-family: 'Segoe UI', sans-serif;
            }
            .stSelectbox label, .stNumberInput label {
                font-weight: 600;
                color: #2e7d32;
            }
        </style>
    """, unsafe_allow_html=True)

# Prediction Logic
def prediction(input):
    country, crop, avg_rain, pesticides, avg_temp = input

    input_df = pd.DataFrame({
        'average_rainfall': [avg_rain],
        'presticides_tonnes': [pesticides],
        'avg_temp': [avg_temp]
    })

    # Create dummy variable structure based on df_final
    input_df1 = df_final.head(1).iloc[:, 3:] * 0  # all-zero structure

    # Activate relevant dummies
    col_country = f'Country_{country}'
    col_crop = f'Item_{crop}'
    missing = []

    for col in [col_country, col_crop]:
        if col in input_df1.columns:
            input_df1[col] = 1
        else:
            missing.append(col)

    if missing:
        st.error(f"Unsupported category: {', '.join(missing)}")
        return None

    final_df = pd.concat([input_df, input_df1], axis=1)

    final_df = final_df.loc[:, ~final_df.columns.duplicated()]  # 🧼 remove any duplicate columns

    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    try:
        test_input = sc.transform(final_df)
        test_input_poly = pf.transform(test_input)
        predict = model.predict(test_input_poly)

        result = round(((predict[0] / 100) * 2.47105), 2)
        return (f"The Production of Crop Yields:- {result} quintel/acers yield Production. "
                f"That means 1 acre of land produces {result} quintel of yield crop.")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Run app
if __name__ == '__main__':
    main()
