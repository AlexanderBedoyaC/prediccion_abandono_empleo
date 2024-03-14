# Librerías para manipulación de datos
import pandas as pd
import numpy as np

# Librerías para visualización
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Librerías para machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#Librerías para la coneción con bigquery
from google.cloud import bigquery
from google.oauth2 import service_account

#Diccionario de datos con nombre y descripción de variables
DATA_DICT = {
    'Age': 'Edad del colaborador',
    'Attrition': 'Si el colaborador se renunció a su empleo o no el año anterior',
    'BusinessTravel': 'Frecuencia con la que los colaboradors viajaron por motivos de trabajo en el último año',
    'Department': 'Departamento en la empresa',
    'DistanceFromHome': 'Distancia del domicilio en kms',
    'Education': 'Nivel de estudios',
    'EducationField': 'Ámbito de formación',
    'EmployeeCount': 'Número de colaboradors',
    'EmployeeID': 'Id de colaborador',
    'Gender': 'Sexo del colaborador',
    'JobLevel': 'Nivel del puesto en la empresa en una escala de 1 a 5',
    'JobRole': 'Nombre del puesto de trabajo en la empresa',
    'MaritalStatus': 'Estado civil del colaborador',
    'MonthlyIncome': 'Ingresos mensuales en dólares al mes',
    'NumCompaniesWorked': 'Número total de empresas en las que ha trabajado el colaborador',
    'Over18': 'Si el colaborador es mayor de 18 años o no',
    'PercentSalaryHike': 'Porcentaje de aumento salarial en el último año',
    'StandardHours': 'Horas estándar de trabajo del colaborador',
    'StockOptionLevel': 'Nivel de opciones sobre acciones del colaborador',
    'TotalWorkingYears': 'Número total de años que el colaborador ha trabajado hasta ahora',
    'TrainingTimesLastYear': 'Número de veces que se impartió formación a este colaborador el año pasado',
    'YearsAtCompany': 'Número total de años que el colaborador lleva en la empresa',
    'YearsSinceLastPromotion': 'Número de años desde el último ascenso',
    'YearsWithCurrManager': 'Número de años bajo el mando del jefe actual',
    'EnvironmentSatisfaction': 'Nivel de satisfacción del entorno de trabajo',
    'JobSatisfaction': 'Nivel de satisfacción laboral',
    'WorkLifeBalance': 'Nivel de conciliación de la vida laboral y familiar',
    'JobInvolvement': 'Nivel de implicación en el trabajo',
    'PerformanceRating': 'Valoración del rendimiento en el último año',
    'MeanTime': 'Tiempo promedio de trabajo al día del colaborador en el último año',
    'retirementType': 'Tipo de retiro',
    'resignationReason': 'Razón de la renuncia',
    'retirementDate': 'Fecha de retiro',
    'mes_renuncia' : 'Mes de renuncia'
}

# Se conecta a bigquery y trae la tabla que se le pase
def get_data(tabla):
    
    credentials_path = 'utils/clave_db.json' #Ruta del archivo json con credenciales de bigquery
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    project_id = 'aplicaciones-analitica' #Nombre del proyecto en bigquery
    client = bigquery.Client(credentials= credentials, project=project_id) # creacion del client para trabajar en python

    query = f'SELECT * FROM recursos_humanos.{tabla}' #Consultar para obtener los datos
    data = client.query(query).to_dataframe() #Obtener tabla completa y convertir en un dataframe
    
    return data

# Genera tabla de frecuencias para una variable
def tabla_frecuencias(df, col):
    
    n = df.shape[0]
    tabla = df.groupby([col])[['EmployeeID']].count().rename(columns={'EmployeeID':'Frecuencia Absoluta'}).reset_index()
    tabla['Frecuencia Relativa'] = tabla['Frecuencia Absoluta'].apply(lambda x: str(round(100*x/n, 3))+' %')
    
    return tabla.sort_values(by='Frecuencia Absoluta', ascending=False)

# visualizar outliers
def vizualizar_outliers(data, col):
    
    fig = make_subplots(rows = 1, cols = 2,
                    subplot_titles = ('Boxplot', 'Histograma'),
                    column_widths = [0.7, 0.3])

    fig.add_trace(
        go.Box(x = data[col]),
        row = 1, col = 1
    )

    fig.add_trace(
        go.Histogram(x = data[col]),
        row = 1, col = 2
    )

    fig.update_layout(title_text = DATA_DICT[col], height=400, width=1000, showlegend = False)
    fig.show()

# Genera tabla de frecuencias y gráfico de barras para una variable
def univariado_barras(df, col, orientation='v'):
    
    if orientation=='v':
        x = col
        y = ['Frecuencia Absoluta']
    else:
        x = ['Frecuencia Absoluta']
        y = col
    
    tabla = tabla_frecuencias(df, col)
    
    fig = px.bar(tabla,
             x = x,
             y = y,
             text_auto = True,
             title = DATA_DICT[col],
             height = 400,
             labels = {'value': 'Total', col:col},
             text = 'Frecuencia Relativa', orientation=orientation)
    fig.layout.update(showlegend=False)
    fig.show()
    
    return tabla

# Genera tabla de frecuencias y gráfico de torta para una variable
def univariado_torta(df, col, hole=0):
    
    tabla = tabla_frecuencias(df, col)
    labels = tabla[col]
    values = tabla['Frecuencia Absoluta']

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,
                                textinfo = 'value+percent',
                                hole = hole
                                )])
    fig.update_layout(
        title_text = DATA_DICT[col],
        height = 400, width=600)
    fig.show()
    
    return tabla

# Genera gráfico de barras vibariado
def analisisBivariado(df, variables, orient, mode, color=px.colors.qualitative.Plotly):
    
    contingency_table=pd.crosstab(df[variables[0]], df[variables[1]])
    contingency_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    
    fig=px.bar(contingency_table,
               orientation = orient,
               barmode = mode,
               color_discrete_sequence = color)
    
    fig.update_layout(width = 800,
                      title = dict(text = DATA_DICT[variables[1]], x=0.5))
    fig.update_traces(texttemplate = '%{value:.2%}', textposition = 'outside')
    
    fig.show()
    print("Tabla de contingencia:")
    
    return contingency_table

# Imputar outliers con IQR
def imputar_outliers(data, cols, th):

    for col in cols:

        q1, q2 = data[col].quantile(0.25), data[col].quantile(0.75)
        iqr = q2 - q1
        lim_inf, lim_sup = q1 - th * iqr, q2 + th * iqr
        data = data[(data[col] >= lim_inf) & (data[col] <= lim_sup)]
        
    return data

# Boxplot y describe 
def bivariado_num(data,variables):
    fig=px.box(data, x=variables[0], y=variables[1],color= variables[0], title=DATA_DICT[variables[1]])
    fig.update_layout(showlegend=False)
    fig.show()
    print('',DATA_DICT[variables[1]], 'de los que no se retiraron: ')
    print(data[data[variables[0]]=='No'][variables[1]].describe())
    print('\n',DATA_DICT[variables[1]],'de los que si se retiraron: ')
    print(data[data[variables[0]]=='Sí'][variables[1]].describe())

#Regresión logística
def regresionLogistica (df,y):
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    
    X_train, X_test,y_train,y_test = train_test_split(df,y,test_size=0.2, random_state=42)

    X_train_std = X_train
    X_test_std = X_test

    lr = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42).fit(X_train_std,y_train)
    y_pred_train = lr.predict(X_train_std)
    y_pred_test = lr.predict(X_test_std)
    y_pred_prob_train = lr.predict_proba(X_train_std)
    y_pred_prob_test = lr.predict_proba(X_test_std)
    mc_train=confusion_matrix(y_train,y_pred_train)
    mc_test=confusion_matrix(y_test,y_pred_test)
    tn, fp, fn, tp = mc_train.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('-'*30,'TRAIN','-'*30)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Especificidad: {especificidad}')
    print(f'F1 score: {f1_score}')
    print('Train score: ',lr.score(X_train_std,y_train))

    tn, fp, fn, tp = mc_test.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    accuracy = lr.score(X_test_std,y_test)
    print('-'*30,'TEST','-'*30)
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'Especificidad : {especificidad}')
    print(f'F1 score : {f1_score}')
    print('Test score: ',accuracy)
    
    resultados = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'y_pred_prob_train' : y_pred_prob_train,
        'y_pred_prob_test' : y_pred_prob_test,
        'precision' : precision,
        'recall' : recall,
        'especificidad' : especificidad,
        'f1_score' : f1_score,
        'accuracy' : accuracy
    }
    
    return resultados

#Bosque aleatorio clasificador
def bosqueAleatorio (df,y):
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    
    X_train, X_test,y_train,y_test = train_test_split(df,y,test_size=0.2, random_state=42)

    num=df.select_dtypes(include=[float,int]).columns


    ranfor=RandomForestClassifier(n_estimators=300,
                                  max_depth=30,
                                  n_jobs=-1,
                                  max_leaf_nodes=20,
                                  min_samples_leaf=10,
                                  class_weight="balanced", random_state=42).fit(X_train,y_train)
    y_pred_train=ranfor.predict(X_train)
    y_pred_test=ranfor.predict(X_test)
    y_pred_prob_train=ranfor.predict_proba(X_train)
    y_pred_prob_test=ranfor.predict_proba(X_test)
    mc_train=confusion_matrix(y_train,y_pred_train)
    mc_test=confusion_matrix(y_test,y_pred_test)
    tn, fp, fn, tp = mc_train.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print('-'*30,'TRAIN','-'*30)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Especificidad: {especificidad}')
    print(f'F1 score: {f1_score}')
    print('Train score: ',ranfor.score(X_train,y_train))

    tn, fp, fn, tp = mc_test.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    accuracy = ranfor.score(X_test,y_test)
    print('-'*30,'TEST','-'*30)
    print(f'Precision : {precision}')
    print(f'Recall : {recall}')
    print(f'Especificidad : {especificidad}')
    print(f'F1 score : {f1_score}')
    print('Test score: ',accuracy)
    
    resultados = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'y_pred_prob_train' : y_pred_prob_train,
        'y_pred_prob_test' : y_pred_prob_test,
        'precision' : precision,
        'recall' : recall,
        'especificidad' : especificidad,
        'f1_score' : f1_score,
        'accuracy' : accuracy
    }
    
    return resultados

def metricas(model, X, y, t):
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    y_pred = model.predict(X)

    mc = confusion_matrix(y, y_pred)
    
    print('-'*30,t,'-'*30)
    
    cm1_display = ConfusionMatrixDisplay(confusion_matrix = mc)
    cm1_display.plot()
    plt.show()

    tn, fp, fn, tp = mc.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    especificidad = tn / (fp + tn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Especificidad: {especificidad}')
    print(f'F1 score: {f1_score}')
    print('Train score: ',model.score(X,y))
    
    return precision, recall, especificidad, f1_score

def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos