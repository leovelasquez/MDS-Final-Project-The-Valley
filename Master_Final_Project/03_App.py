


import streamlit as st
import pandas as pd
import joblib
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter



# Load the trained Random Forest Regressor model
model = joblib.load("/content/drive/MyDrive/MDS_Final-Project _Leo/01_Notebooks_Field-Work/Modelo_Final/01_RF_model_R2-85_Test_Leo.pkl")



# -- Set page config
apptitle = 'AI Idealista'
st.set_page_config(page_title=apptitle, page_icon=":soon:")


# Streamlit app title
st.title("AI Idealista")


st.markdown("""
  La información de tu inmueble aparecerá aquí después de que hayas introducido los datos de tu inmueble
""")


#Load the dataframe of the neighborhoods
@st.cache_data
def load_data():
    return pd.read_csv("/content/drive/MyDrive/MDS_Final-Project _Leo/00_Data-sets/df_Barrios_produccion.csv")

df = load_data()

#Create the list of neighborhoods to invoke it in the neighborhoods input widget later on
barrios_list = df["LOCATIONNAME"].tolist()


# Function to retrieve DISTANCE_TO_CITY_CENTER_M for a given neighborhood
def get_CITY_CENTER(BARRIO):
    try:
        CITY_CENTER = df.loc[df['LOCATIONNAME'] == BARRIO, 'DISTANCE_TO_CITY_CENTER_M'].values[0]
        return CITY_CENTER
    except IndexError:
        return None


# Function to retrieve DISTANCE_TO_CASTELLANA_M for a given neighborhood
def get_CASTELLANA(BARRIO):
    try:
        CASTELLANA = df.loc[df['LOCATIONNAME'] == BARRIO, 'DISTANCE_TO_CASTELLANA_M'].values[0]
        return CASTELLANA
    except IndexError:
        return None


# Function to retrieve DISTANCE_TO_METRO_M for a given neighborhood
def get_METRO(BARRIO):
    try:
        METRO = df.loc[df['LOCATIONNAME'] == BARRIO, 'DISTANCE_TO_METRO_M'].values[0]
        return METRO
    except IndexError:
        return None

# Function to retrieve LOCATION_MEAN_UNITPRICE_log for a given neighborhood
def get_LOCATION_MEAN(BARRIO):
    try:
        LOCATION_MEAN = df.loc[df['LOCATIONNAME'] == BARRIO, 'LOCATION_MEAN_UNITPRICE_log'].values[0]
        return LOCATION_MEAN
    except IndexError:
        return None


# Function to retrieve R2_Cuartiles for a given neighborhood
def get_R2_CUARTILES(BARRIO):
    try:
        R2_CUARTILES = df.loc[df['LOCATIONNAME'] == BARRIO, 'R2_Cuartiles'].values[0]
        return R2_CUARTILES
    except IndexError:
        return None


# Function to transform selectbox values to 1 or 0
def transform_selectbox_value(value):
    return 1 if value == "Si" else 0


# Collect input variables

st.sidebar.image("/content/drive/MyDrive/MDS_Final-Project _Leo/01_Notebooks_Field-Work/Archivos_producción/idealista.jpg")
st.sidebar.markdown('# Cuentános más sobre tu vivienda')
st.sidebar.markdown("""
 * Ingresa la información sobre tu Inmueble y haz click en PREDICT
 * Si no conoces un valor, no te preocupes, deja el valor que está por default
""")


#Input widgets

st.sidebar.markdown('## :clipboard: Datos Generales')
CONSTRUCTEDAREA = st.sidebar.number_input('¿Cuántos(m²) tiene tu inmueble?', min_value=30, max_value=400, step=1, value=150)
ROOMNUMBER = st.sidebar.number_input('¿Cuántas habitaciones tiene tu inmueble?', min_value=1, max_value=10, step=1, value=3)
BATHNUMBER = st.sidebar.number_input('¿Cuántos baños tiene tu inmueble? ', min_value=1, max_value=10, step=1, value=3)
HASLIFT = transform_selectbox_value(st.sidebar.selectbox("¿Tu finca tiene ascensor?", ["Si", "No"]))
ISDUPLEX = transform_selectbox_value(st.sidebar.selectbox("¿Es Duplex?", ["Si", "No"]))
CADCONSTRUCTIONYEAR = st.sidebar.slider('Año de construcción de tu finca, si no sabes deja 1985', min_value=1500, max_value=2023, step=1, value=1985)
st.sidebar.markdown('## :house_with_garden: Tipo de vivienda')
BUILTTYPEID_1 = transform_selectbox_value(st.sidebar.selectbox("¿Tu inmueble es una construcción nueva?", ["Si", "No"]))
BUILTTYPEID_2 = transform_selectbox_value(st.sidebar.selectbox("¿Tu inmueble es de segunda mano sin remodelar?", ["Si", "No"]))
BUILTTYPEID_3 = transform_selectbox_value(st.sidebar.selectbox("¿Tu inmueble es de segunda mano en buenas condiciones?", ["Si", "No"]))

st.sidebar.markdown('## :round_pushpin: Ubicación')
BARRIO = st.sidebar.selectbox("Pick one", barrios_list)

city = st.sidebar.text_input("City", "Madrid")
province = st.sidebar.text_input("Province", "Madrid")
street = st.sidebar.text_input("Street", "Gran via 84")
country = st.sidebar.text_input("Country", "Spain")

DISTANCE_TO_CITY_CENTER_G = get_CITY_CENTER(BARRIO)
DISTANCE_TO_METRO_G = get_METRO(BARRIO)
DISTANCE_TO_CASTELLANA_G = get_CASTELLANA(BARRIO)
LOCATION_MEAN_G = get_LOCATION_MEAN(BARRIO)


R2_CUARTILES_G = get_R2_CUARTILES(BARRIO)


# Predict when the "Predict" button is clicked
if st.sidebar.button("Predict"):
    # Create a DataFrame from the collected inputs
    data = {
        'CONSTRUCTEDAREA': [CONSTRUCTEDAREA],
        'ROOMNUMBER': [ROOMNUMBER],
        'BATHNUMBER': [BATHNUMBER],
        'HASLIFT': [HASLIFT],
        'ISDUPLEX': [ISDUPLEX],
        'CADCONSTRUCTIONYEAR': [CADCONSTRUCTIONYEAR],
        'BUILTTYPEID_1': [BUILTTYPEID_1],
        'BUILTTYPEID_2': [BUILTTYPEID_2],
        'BUILTTYPEID_3': [BUILTTYPEID_3],
        'DISTANCE_TO_CITY_CENTER': [DISTANCE_TO_CITY_CENTER_G],
        'DISTANCE_TO_METRO': [DISTANCE_TO_METRO_G],
        'DISTANCE_TO_CASTELLANA': [DISTANCE_TO_CASTELLANA_G],
        'LOCATION_MEAN_UNITPRICE_log': [LOCATION_MEAN_G],
    }

    # Initialize variables
    lat = None
    lon = None


    input_df = pd.DataFrame(data)

    # Make predictions
    prediction_log = model.predict(input_df)

    # Transform the log prediction to exponential scale
    prediction_exp = np.exp(prediction_log)
    prediction_formated = format(prediction_exp[0], ".0f")


    # Calculate the estimated house price
    estimated_price = prediction_exp * CONSTRUCTEDAREA

    R2_CUARTILES_G_X = R2_CUARTILES_G


    # Display the estimated total price
    estimated_price_float = float(estimated_price)
    st.write("### Estimación del precio total:")
    st.markdown(f"<h1 style='text-align:center;color:#F63366;'>€ {estimated_price_float:.0f}</h1>", unsafe_allow_html=True)


    # Display the prediction
    st.write("### El valor del m2 de tu vivienda es:")
    st.markdown(f"<h2 style='text-align:center;'>€ {prediction_formated}</h2>", unsafe_allow_html=True)


    # Display the confidence mark
    st.write("### Nivel de confianza:")
    st.markdown(f"<h2 style='text-align:center;'>{R2_CUARTILES_G_X}</h2>", unsafe_allow_html=True)



    # Display the collected data in a DataFrame
    st.write("Collected Data:")
    st.dataframe(input_df)


    #Display the map and location of the property

    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street + ", " + city + ", " + province + ", " + country)

    if location:
        lat = location.latitude
        lon = location.longitude

    # Display the map in the main content area if lat and lon are available
    if lat is not None and lon is not None:
      map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
      st.map(map_data, zoom=12)

    st.markdown("""
    ### Estos son algunos atributos extra que pueden influenciar positiva o negativamente el precio de tu inmueble:
     * El inmueble dispone de cédula de habitabilidad
     * El inmueble tiene climatización
     * El inmueble tiene terraza o balcones
     * El inmueble tiene buena vista
     * El inmueble tiene ilumicación natural
    """)

    st.markdown("""
    ### Sube la vivienda al portal inmobiliario de idealista
    Idealista es el mayor escaparate inmobiliario de España. Según datos de SimilarWeb en el mes de diciembre de 2022 el portal tuvo más de 43 millones de visitas, muy por encima de sus competidores. Si quieres que tu piso llegue a todos los compradores posibles, no puedes omitir este paso.
    Además, la agencia inmobiliaria que te acompañe en el proceso será la encargada de gestionar todas las visitas del inmueble y así, vender la vivienda en tiempo récord.
    """)




st.subheader("Acerca de esta app")
st.markdown("""
Esta app ha sido desarrollada como trabajo final del MDS de The Valley por Alejandro López y Leonardo Velásquez bajo la dirección de David Rey, CDO de Idealista.


Para mas información [see the code](https://github.com/leovelasquez/MDS-Final-Project).

""")

