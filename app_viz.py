import streamlit as st 

import numpy as np 
import pandas as pd  

import matplotlib.pyplot as plt  
import plotly.graph_objects as go



# path data
PATH_DATA = 'data/diabetes.csv'

# functions

# mascara para categortizacion
dict_mask = {
    'Pregnancies':{
        'Preg 0': [0],
        'Preg 1': [1],
        'Preg 2': [2],
        'Preg 3': [3],
        'Preg 4+':[4, 100]
    },
    'Glucose': {
        'Glu 0-50': [0, 50],
        'Glu 51-100': [50, 100],
        'Glu 101-150': [100, 150],
        'Glu >150': [150, 500]
    },
    'BloodPressure': {
        'BP 0-50':[0, 50],
        'BP 50-100': [50, 100],
        'BP >100': [100, 500]
    },
    'SkinThickness': {
        'SkinT 0-25': [0, 25],
        'SkinT 26-50': [25, 50],
        'SkinT 51-75': [50, 75],
        'SkinT >75': [75, 500]     
    },
    'Insulin': {
        'Ins 0-25': [0, 25],
        'Ins 26-50': [25, 50],
        'Ins 51-75': [50, 75],
        'Ins >75': [75, 5000]    
    },
    'BMI': {
        'BMI 0-25': [0, 25],
        'BMI 25.01-50': [25, 50],
        'BMI >50': [50, 500]
    },
    'DiabetesPedigreeFunction': {
        'DPF 0-0.5': [0, 0.5],
        'DPF 0.51-1.00': [0.5, 1.00],
        'DPF >1.00': [1, 100] 
    },
    'Age': {
        'age 0-20': [0, 20],
        'age 21-40': [20, 40],
        'age 41-60': [40, 60],
        'age 61-80': [60, 80],
        'age >80':[80, 500]
    },
    'Outcome': {
        'no': [0],
        'yes': [1]
    }
}


def run_mask(value, var_name, dict_mask):
    # print(var_name, value)
    for key, values in dict_mask[var_name].items():
        # print(key, values)
        if len(values) > 1:
            if values[0] <= value < values[1]:
                masked_value = key
                # print("masked_value", masked_value)
        else:
            if value == values[0]:
                masked_value = key
                # print("masked_value", masked_value)

    return masked_value


def build_masked_df(df, dict_mask):

    df_mask = df.copy()

    for variable in df_mask.columns:
        df_mask[variable] = df_mask[variable].apply(
            lambda x: run_mask(x, variable, dict_mask)
        )
    df_mask['index'] = df_mask.index

    return df_mask


def build_data_links(df_mask):

    data_links = []
    columns = [variable for variable in df_mask.columns if variable != 'index']

    for i in range(len(columns) - 1):
        # print(columns[i], columns[i+1])
        source_column = columns[i]
        target_column = columns[i + 1]

        df_count = df_mask.groupby([source_column, target_column])['index'].count().reset_index()
        df_count.columns = ['source', 'target', 'value']
        # print(df_count)
        data_links.append(df_count)

    data_link_df = pd.concat(data_links)

    return data_link_df


def sankey_plot(data_link_df):

    unique_source_target = list(pd.unique(data_link_df[['source', 'target']].values.ravel('K')))

    #for assigning unique number to each source and target
    mapping_dict = {k: v for v, k in enumerate(unique_source_target)}

    #mapping of full data
    data_link_df['source'] = data_link_df['source'].map(mapping_dict)
    data_link_df['target'] = data_link_df['target'].map(mapping_dict)

    #converting full dataframe as list for using with in plotly
    links_dict = data_link_df.to_dict(orient='list')

    #Sankey Diagram Code 
    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = unique_source_target,

        ),
        link = dict(
        source = links_dict["source"],
        target = links_dict["target"],
        value = links_dict["value"],

    ))])

    fig.update_layout(title_text="", font_size=10,width=1000, height=600)
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)


def gauge_plot(df):
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = df['Age'].mean(),
        mode = "gauge+number",
        title = {'text': "Mean Age Diabetes Study"},
        
        gauge = {'axis': {'range': [None, 100]},
                'steps' : [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 40], 'color': "gray"},
                    {'range': [40, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgray"}
                    ],
                }))

    # fig.show()
    st.plotly_chart(fig, use_container_width=True)


def plot_hexbin_0(df):

    # fig = df[df['Outcome'] == 0].plot.hexbin(x='Age', y='BMI', gridsize=25, title="Age - BMI relation, no Diabetes")
    fig, ax = plt.subplots()

    x = df[df['Outcome'] == 0]['Age']
    y = df[df['Outcome'] == 0]['BMI']

    ax.hexbin(x = x, y = y, gridsize = 15,
            edgecolor = "white", linewidths = 1.5)
    plt.xlabel("Age")
    plt.ylabel("BMI")
    plt.title(' Age vs BMI, no diabetes')
    # fig.show()
    st.pyplot(fig)

def plot_hexbin_1(df):
    
    # fig = df[df['Outcome'] == 0].plot.hexbin(x='Age', y='BMI', gridsize=25, title="Age - BMI relation, no Diabetes")
    fig, ax = plt.subplots()
    x = df[df['Outcome'] == 1]['Age']
    y = df[df['Outcome'] == 1]['BMI']
    
    ax.hexbin(x = x, y = y, gridsize = 15,
            edgecolor = "white", linewidths = 1.5)
    plt.xlabel("Age")
    plt.ylabel("BMI")
    plt.title(' Age vs BMI, with diabetes')
    # fig.show()
    st.pyplot(fig)


def create_graph(graph_type, df):  

    if graph_type == "Sankey diagram":

        st.subheader("Diagrama de Sankey")
        st.markdown("""
                    En esta visualización se ha representado la frecuencia de diferentes categorias 
                    de variables y su relación con las demás. Para ello, dado que el conjunto de datos lo componen
                    variables numéricas, se ha aplicado una discretización para asignar un conjunto de categorías
                    en base a unos invervalos definidos.
        """)
        df_mask = build_masked_df(df, dict_mask)
        data_link_df = build_data_links(df_mask)
        
        with st.container():  
            sankey_plot(data_link_df)

    elif graph_type == "Gauge Plot":  
        with st.container():  
            st.subheader("Gauge Plot")
            st.markdown("""
                    Para poder emplear un gráfico Gauge, se ha representado a modo de ejemplo el valor medio 
                    de las personas que han participado en este estudio. Este gráfico también representa los
                    diferentes intervalos que se han empleado en la discretización.
            """)
            gauge_plot(df)

    elif graph_type == "Hexbin Plot": 
        
        st.subheader("Hexbin Plot")
        st.markdown("""
                    En este caso se ha empleado un hexbin plot para representar la relación entre 
                    la edad y el Indice de Masa Corporal. Para una mayor distinción, se ha separado
                    en dos gráficas los grupos con diabetes y sin diabetes.
        """)

        with st.container():
            plot_hexbin_0(df)
        with st.container():
            plot_hexbin_1(df)

    else:  
        st.write("Selecciona una gráfica en el menú lateral")  





# sidebar 
st.sidebar.title("Selección de visualización")  
graph_type = st.sidebar.selectbox(  
    "Tipo de gráfica",  
    ("Sankey diagram", "Gauge Plot", "Hexbin Plot")  
)
st.sidebar.header("Dataset empleado")
st.sidebar.subheader('Pima Indians Dataset')
st.sidebar.markdown('''
Este conjunto de datos procede del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales.
El objetivo del conjunto de datos es **predecir diagnósticamente si un paciente tiene o no diabetes**, 
basándose en determinadas mediciones diagnósticas incluidas en el conjunto de datos.
En particular, todos los pacientes son mujeres de al menos 21 años y de ascendencia india pima.

El dataset consta de varias variables médicas predictoras y una variable objetivo, ``Outcome``. 
Las variables predictoras incluyen el número de embarazos que ha tenido la paciente, 
su IMC, nivel de insulina, edad, etc.
''')
st.sidebar.link_button("Dataset en Kaggle", "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")

# main
st.title('Visualización de datos - PEC 2')
st.subheader("José Luis Rodriguez Andreu")
st.write("Master Universitario en Ciencia de Datos")


df = pd.read_csv('data/diabetes.csv')  
create_graph(graph_type, df)  