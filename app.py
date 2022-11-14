from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import math
import random

from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio import start_server


import re
import numpy as np
from nltk.tokenize import word_tokenize as wt 
#from nltk.stem.porter import PorterStemmer
#stemmer = PorterStemmer()
import joblib
from scipy import spatial
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)

def proses_dil(dil):
    dil[['KOORDINAT_X','KOORDINAT_Y']]=dil[['KOORDINAT_X','KOORDINAT_Y']].fillna(0)
    dil_error=dil[dil[['KOORDINAT_X','KOORDINAT_Y']].apply(pd.to_numeric,errors='coerce').isnull().any(axis=1)].reset_index(drop=True)
    dil_valid=dil[dil[['KOORDINAT_X','KOORDINAT_Y']].apply(pd.to_numeric,errors='coerce').notnull().all(axis=1)].reset_index(drop=True)
    dil_valid[['KOORDINAT_X','KOORDINAT_Y']]=dil_valid[['KOORDINAT_X','KOORDINAT_Y']].astype(float)
    dil_ok=dil_valid[(dil_valid.KOORDINAT_X != 0)|(dil_valid.KOORDINAT_X != 0)].reset_index(drop=True)
    dil_null=dil_valid[(dil_valid.KOORDINAT_X == 0)&(dil_valid.KOORDINAT_X == 0)].reset_index(drop=True)
    #print (str(dil_ok.shape(0))+"-"+str(dil_null.shape(0))+"-"+str(dil_error.shape(0)))
    return dil_valid, dil_ok, dil_null, dil_error

def classify(X,y,nama): 
    # split train and test data  
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #Classifier
    classifier_multi = MultinomialNB(alpha=0.01)
    np.seterr(divide = 'ignore')

    batch_size=10000 # Jml batch sample, turunkan nilai ini jika PC RAM nya kecil
    n=X_train.shape[0]/batch_size
    n=int(math.ceil(n))

    X_train_splitted=[]
    #X_train_splitted = np.array_split(X_train, batch_size)
    #X_train_splitted = [X_train[i:i + batch_size] for i in range(0, len(X_train.shape[0]), batch_size)]
    for i in range(n):
        X_train_splitted.append(X_train[i*batch_size:i*batch_size+batch_size])
    #X_train_splitted.append(X_train[i*batch_size+batch_size:])

    y_train_splitted=[]
    #y_train_splitted = np.array_split(y_train, batch_size)
    #y_train_splitted = [y_train[i:i + batch_size] for i in range(0, len(y_train.shape[0]), batch_size)]
    for i in range(n):
        y_train_splitted.append(y_train[i*batch_size:i*batch_size+batch_size])
    #y_train_splitted.append(y_train[i*batch_size+batch_size:])

    put_text("Metode : Multinomial Naive Bayes").style('text-align:center'),

    for i in range(len(X_train_splitted)):
        classifier_multi.partial_fit(X_train_splitted[i], y_train_splitted[i],y.unique())

        # predict class & accuracy
        y_pred_multi = classifier_multi.predict(X_test)
        accuracy_multi = accuracy_score(y_test, y_pred_multi)
        #put_text("prediksi")
        put_text("Akurasi Training batch sample ke "+str(i+1)+" :"+str(accuracy_multi*100)).style('text-align:center'),

    #joblib_file_MNB = "joblib_MNB_Model_"+str(nama)+".pkl"
    joblib_file_MNB = "joblib_MNB_Model_GARDU.pkl"  
    joblib.dump(classifier_multi, joblib_file_MNB)
    #put_text("joblib OK")

def sebar(radius_sebaran):
	n=(math.sqrt(random.random())*math.cos(math.radians(random.randint(0, 360)))*radius_sebaran/111319)
	return n

def predict():
    put_markdown("# Perbaikan Data Koordinat Pelanggan").style('text-align:center'),
    put_markdown("## menggunakan Machine Learning").style('text-align:center; color:SteelBlue'),
    file=input_group(
        'Data DIL unit',
        [file_upload('Data DIL :',name='dil', accept='.csv', multiple=False),
         file_upload('Data GARDU :',name='gardu', accept='.csv', multiple=False)])

    open(str(file['dil']['filename']), 'wb').write(file['dil']['content'])
    data_dil = pd.read_csv(str(file['dil']['filename']))
    dil_valid, dil_ok, dil_null, dil_error=proses_dil(data_dil)
    #put_text(str(dil_ok.shape[0])+"-"+str(dil_null.shape[0])+"-"+str(dil_error.shape[0]))

    open(str(file['gardu']['filename']), 'wb').write(file['gardu']['content'])
    data_gd = pd.read_csv(str(file['gardu']['filename']))
    
    fig_ori = go.Figure()
    fig_ori.add_trace(go.Scattermapbox(lat = dil_ok['KOORDINAT_X'],
                        lon = dil_ok['KOORDINAT_Y'],
                        mode = 'markers+text',
                        marker={'color':'rgb(70, 120, 227)'},
                        name ='Koordinat OK ('+str(dil_ok.shape[0])+")",
                        text = "<br>"+dil_ok['IDPEL'].astype(str)+" - "+dil_ok['NAMA'].astype(str)+"<br>"+dil_ok['NAMAPNJ'].astype(str),
                        textposition="top center"))
    
    fig_ori.add_trace(go.Scattermapbox(lat = dil_null['KOORDINAT_X'],
                        lon = dil_null['KOORDINAT_Y'],
                        mode = 'markers+text',
                        marker={'color':'rgb(227, 70, 128)'},
                        name ='Koordinat Null ('+str(dil_null.shape[0])+")",
                        text = "<br>"+dil_null['IDPEL'].astype(str)+" - "+dil_null['NAMA'].astype(str)+"<br>"+dil_null['NAMAPNJ'].astype(str),
                        textposition="top center"))

    fig_ori.add_trace(go.Scattermapbox(lat = data_gd['LATITUDE'],
                        lon = data_gd['LONGITUDE'],
                        mode = 'markers+text',
                        marker={'color':'rgb(78, 198, 103)','size':10},
                        name ='Gardu ('+str(data_gd.shape[0])+")",
                        text = "<br>"+data_gd['NAMA GARDU'].astype(str)+" - "+data_gd['NOMOR GARDU'].astype(str)+"<br>"+data_gd['ALAMAT'].astype(str),
                        textposition="top center"))
    fig_ori.update_layout(mapbox_style="open-street-map")
    fig_ori.update_mapboxes(center={'lat':-4.62909,'lon':122.0977},zoom=6)
    fig_ori.update_layout(title='Data DIL - Original',title_x=0.5,title_y=0.85)
    html_ori = fig_ori.to_html(include_plotlyjs="require", full_html=False)
    put_html(html_ori)

    param_nlp=input_group(
        'Parameter NLP',
        [slider('Silahkan input jarak valid maksimum titik pelanggan ke Gardu (meter)',name='jarak_valid',min_value=300,max_value=2000,step=50),
         slider('Silahkan input max-radius sebaran titik pelanggan (meter)',name='radius_sebaran',min_value=50,max_value=500,step=10)])

    put_markdown('Validity Threshold').style('text-align:center'),
    put_markdown(str(param_nlp['jarak_valid'])+' meter').style('text-align:center; color:SteelBlue; font-weight: bold'),

    put_markdown('Prediction Max-Radius').style('text-align:center'),
    put_markdown(str(param_nlp['radius_sebaran'])+' meter').style('text-align:center; color:SteelBlue; font-weight: bold'),

    #====================== Find Nearest Neigbour =========================
    #define point coordinate columns
    koordinat_dil = dil_valid[["KOORDINAT_X", "KOORDINAT_Y"]]
    koordinat_gd = data_gd[["LATITUDE", "LONGITUDE"]]
    
    # build kdtree DIL
    kdtree_gd = spatial.cKDTree(koordinat_gd)

    # query tree DIL to GD coordinates. NOTICE the k=1 nearest neigborgh
    distances, indexes = kdtree_gd.query(koordinat_dil, k=1)
    distances=distances*110950 #convert degree to meters

    # assign it to a new dataframe
    new_dil_valid = dil_valid.assign(Closest_gd=data_gd["NOMOR GARDU"][indexes].array)
    new_dil_valid = new_dil_valid.assign(Latitude_gd=data_gd["LATITUDE"][indexes].array)
    new_dil_valid = new_dil_valid.assign(Longitude_gd=data_gd["LONGITUDE"][indexes].array)
    new_dil_valid = new_dil_valid.assign(Closest_gd_Dist=distances)

    #====================== ML Process NLP =========================
    #Load dataset
    training_dataset=new_dil_valid[new_dil_valid.Closest_gd_Dist < param_nlp['jarak_valid']].reset_index() #Jarak < parameter misal 1000m valid
    test_dataset=new_dil_valid[new_dil_valid.Closest_gd_Dist >= param_nlp['jarak_valid']].reset_index() #Jarak >= parameter misal 1000m invalid

    dataset=training_dataset[['Closest_gd','NAMAPNJ']] 
    data = []
    for i in range(dataset.shape[0]):
        alamat = dataset.iloc[i, 1]

        # remove non alphabatic characters
        alamat = re.sub('[^A-Za-z]', ' ', alamat)

        # make words lowercase, because Go and go will be considered as two words
        alamat = alamat.lower()

        # tokenising
        tokenized_alamat = wt(alamat)

        # remove generic words
        generic_words=["btn",
                        "blok",
                        "lrg",
                        "kel",
                        "blk",
                        "iii",
                        "desa",
                        "kec",
                        "dusun",
                        "the"]
    
        alamat_processed = []
        for word in tokenized_alamat:
            if word not in generic_words:
                alamat_processed.append(word)
                
        alamat_text = " ".join(alamat_processed)
        data.append(alamat_text)

    # creating the feature matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    matrix = TfidfVectorizer(analyzer='word',
                            token_pattern=r'\b[a-zA-Z]{3,}\b',
                            ngram_range=(1, 1),
                            max_features=2000
                            )
    X = matrix.fit_transform(data)
    y = dataset.iloc[:, 0]
    
    # Save the vectorizer
    joblib_file_vectorizer = "joblib_file_vectorizer.pkl"  
    joblib.dump(matrix, joblib_file_vectorizer)

    #put_text("=======================================================")
    put_text("============ CLASSIFY KODE GARDU BY ALAMAT ============").style('text-align:center'),
    #put_text("=======================================================")
    classify(X,y,"GARDU")

    put_text("")
    #put_text("=======================================================")
    put_text("Proses Prediksi Koordinat Invalid Jarak gardu >= "+str(param_nlp['jarak_valid'])+"m").style('text-align:center'),
    #put_text("=======================================================")

    #Predict New DATA
    newdata=test_dataset[['NAMAPNJ']]
    #newdata = pd.read_csv('new_data.csv', encoding='ISO-8859-1');
    data_baru = []
    for i in range(newdata.shape[0]):
        alamat_baru = newdata.iloc[i, 0]
        # remove non alphabatic characters
        alamat_baru = re.sub('[^A-Za-z]', ' ', alamat_baru)
        # make words lowercase, because Go and go will be considered as two words
        alamat_baru = alamat_baru.lower()
        # tokenising
        tokenized_alamat_baru = wt(alamat_baru)
        alamat_baru_text = " ".join(tokenized_alamat_baru)
        data_baru.append(alamat_baru_text)    

    vectorizer = joblib.load("joblib_file_vectorizer.pkl")
    X_new = vectorizer.transform(data_baru).toarray()

    joblib_MNB_Model_GD = joblib.load("joblib_MNB_Model_GARDU.pkl")
    new_y_pred_multi_GD = joblib_MNB_Model_GD.predict(X_new)

    test_dataset=test_dataset.assign(GARDU_pred=new_y_pred_multi_GD)
    test_dataset=test_dataset.assign(keterangan="NOT valid >="+str(param_nlp['jarak_valid'])+'m')
    test_dataset = pd.merge(test_dataset, data_gd[["NOMOR GARDU","LATITUDE","LONGITUDE"]], how="left", left_on='GARDU_pred', right_on='NOMOR GARDU')
    test_dataset = test_dataset.rename(columns={'LATITUDE': 'X_new', 'LONGITUDE': 'Y_new'})
    test_dataset = test_dataset.drop('NOMOR GARDU', axis=1)
    test_dataset['X_new'] = test_dataset['X_new'].astype(float).apply(lambda x: x + sebar(int(param_nlp['radius_sebaran'])))
    test_dataset['Y_new'] = test_dataset['Y_new'].astype(float).apply(lambda x: x + sebar(int(param_nlp['radius_sebaran'])))

    #test_dataset.to_csv("dil_bombana_testOK_v1.csv", index=False)

    training_dataset=training_dataset.assign(GARDU_pred=training_dataset['Closest_gd'])
    training_dataset=training_dataset.assign(keterangan="Valid <"+str(param_nlp['jarak_valid'])+'m')
    training_dataset=training_dataset.assign(X_new=training_dataset["KOORDINAT_X"])
    training_dataset=training_dataset.assign(Y_new=training_dataset["KOORDINAT_Y"])

    hasil_df = pd.concat([training_dataset,test_dataset], ignore_index=True)

    #out to csv
    hasil_df.to_csv(str(file['dil']['filename'])[:-4]+'_perbaikan.csv', index=False)
    #hasil_df.to_csv("dil_bombana_testOK_pred.csv", index=False)

    put_text("=============== Done ==============").style('text-align:center'),

    fig = px.scatter_mapbox(hasil_df,
                        lat="X_new", 
                        lon="Y_new", 
                        color="keterangan", 
                        #size=12,
                        labels="NAMAPNJ", 
                        hover_name="NAMAPNJ",
                        title="Peta DIL - Perbaikan")
    fig.add_trace(go.Scattermapbox(lat = data_gd['LATITUDE'],
                        lon = data_gd['LONGITUDE'],
                        mode = 'markers+text',
                        marker={'color':'rgb(78, 198, 103)','size':10},
                        #marker={'symbol':'triangle'},
                        name ='Gardu',
                        text = "<br>"+data_gd['NAMA GARDU'].astype(str)+" - "+data_gd['NOMOR GARDU'].astype(str)+"<br>"+data_gd['ALAMAT'].astype(str),
                        textposition="top center"))
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(title_x=0.5,title_y=0.85)
    html = fig.to_html(include_plotlyjs="require", full_html=False)
    put_html(html)
    
    #Download Button
    #put_text("Data perbaikan koordinat dapat di download di sini :").style('text-align:center'),
    #put_file(str(file['dil']['filename'])[:-4]+'_perbaikan_download.csv', hasil_df.reset_index(drop=True), 'download')


#if __name__ == '__main__':
#    start_server(app, port=80)

app.add_url_rule('/tool', 'webio_view', webio_view(predict),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)
