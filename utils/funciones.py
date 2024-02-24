def get_data(tabla):
    
    #Librerías para la coneción con bigquery
    from google.cloud import bigquery
    from google.oauth2 import service_account
    
    credentials_path = 'utils/clave_db.json' #Ruta del archivo json con credenciales de bigquery
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    project_id = 'aplicaciones-analitica' #Nombre del proyecto en bigquery
    client = bigquery.Client(credentials= credentials, project=project_id) # creacion del client para trabajar en python

    query = f'SELECT * FROM recursos_humanos.{tabla}' #Consultar para obtener los datos
    data = client.query(query).to_dataframe() #Obtener tabla completa y convertir en un dataframe
    
    return data
    