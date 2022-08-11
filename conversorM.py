import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn import metrics
import numpy as np 
 

st.title('Superliga de Vôlei - Previsão Resultados')
uploader = st.file_uploader("Escolha o Arquivo")




def predction(df_jogos):    
       

    df_final = df_jogos

    df_final['total atq home'] = df_final['total atq home'] / df_final['qtd sets']
    df_final['pontos ataques home'] = df_final['pontos ataques home'] / df_final['qtd sets']
    df_final['atq error home'] = df_final['atq error home'] / df_final['qtd sets']
    df_final['bloqueio home'] = df_final['bloqueio home'] / df_final['qtd sets']
    df_final['server error home'] = df_final['server error home'] / df_final['qtd sets']
    df_final['total atq away'] = df_final['total atq away'] / df_final['qtd sets']
    df_final['pontos ataques away'] = df_final['pontos ataques away'] / df_final['qtd sets']
    df_final['atq error away'] = df_final['atq error away'] / df_final['qtd sets']
    df_final['bloqueio away'] = df_final['bloqueio away'] / df_final['qtd sets']
    df_final['server error away'] = df_final['server error away'] / df_final['qtd sets']
    
    home_team = st.selectbox("Escolha o time da Casa", df_final['team home'])
    away_team = st.selectbox("Escolha o time da Casa", df_final['team away'])

    
    X = df_final.drop(columns=['team home', 'team away','qtd sets', 'vencedor','id_jogo'])
    y = df_final['vencedor']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    model = RandomForestClassifier(min_samples_leaf=5, min_samples_split=2,n_estimators=100)
    model.fit(X_train, y_train)
    estimator = model.estimators_[5]


    y_pred = model.predict(X_test)

    scores = cross_val_score(model, X, y, cv=2)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Score: " , scores.mean())


    df_splitted_away = df_final[['team away','total atq away','pontos ataques away','atq error away','% atq away','% rec away','bloqueio away','server error away','qtd sets','vencedor']]
    cols = ['team', 'total atq', 'pontos ataque', 'atq error', '% atq', '% rec', 'bloqueio', 'server error','qtd sets', 'vencedor']
    df_splitted_away.columns = cols
    df_splitted2 = df_final[['team home','total atq home','pontos ataques home','atq error home','% atq home','% rec home','bloqueio home','server error home','qtd sets','vencedor']]
    df_splitted2.columns = cols   

    df_splitted_away['vencedor'] = ~df_splitted_away['vencedor']

    frames = [df_splitted_away, df_splitted2]
    df_merged2 = pd.concat(frames)

    home = df_merged2.loc[df_merged2['team'] == home_team]
   
    home_medias = [round(home['total atq'].mean()),round(home['pontos ataque'].mean()),round(home['atq error'].mean()),round(home['% atq'].mean()),round(home['% rec'].mean()),round(home['bloqueio'].mean()),round(home['server error'].mean())]
    
    away = df_merged2.loc[df_merged2['team'] == away_team]
  
    away_medias = [round(away['total atq'].mean()),round(away['pontos ataque'].mean()),round(away['atq error'].mean()),round(away['% atq'].mean()),round(away['% rec'].mean()),round(away['bloqueio'].mean()),round(away['server error'].mean())]
   
    resultado =  home_medias + away_medias
    resultado_final = np.array(resultado)
    
   
    import warnings

    warnings.filterwarnings("ignore")
    resultado_chance = model.predict_proba(resultado_final.reshape(1,-1))
    resultado_home_porcetagem = round(resultado_chance[0][1] * 100, 2)
    resultado_away_porcetagem =  round(resultado_chance[0][0] * 100, 2)
    mostrar_resultado = st.button("Calcular Porcentagem")
    if mostrar_resultado:
        st.write("Chances do time " + home['team'].iloc[0] + " vencer: " + str(resultado_home_porcetagem) + '%\n' + " Chances do time " + away['team'].iloc[0] + " vencer: " + str(resultado_away_porcetagem) + "%")
       
        resultado_previsao = "Chances do time " + home['team'].iloc[0] + " vencer: " + str(resultado_home_porcetagem) + '%\n' + " Chances do time " + away['team'].iloc[0] + " vencer: " + str(resultado_away_porcetagem) + "%"    
        st.download_button("Baixar Resultado", resultado_previsao)
    
    


if uploader is not None:
    df_jogos = pd.DataFrame()
    df_jogos = pd.read_csv(uploader)
    st.write(df_jogos)
    predction(df_jogos)
    