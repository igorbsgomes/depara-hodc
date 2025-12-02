# streamlit_app.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import streamlit as st
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Classificador de Grupo/Subgrupo", layout="centered")
st.title("🤖 Classificador Automático de Grupo e Subgrupo (com Avaliação)")

# Upload dos arquivos
st.sidebar.header("Passo 1: Upload dos Arquivos")
historico_file = st.sidebar.file_uploader("Histórico com Grupo/Subgrupo", type=["xlsx"])
novos_file = st.sidebar.file_uploader("Novos registros para classificar", type=["xlsx"])

salvar_historico = st.sidebar.checkbox("Salvar novos dados no histórico? (Gera novo arquivo)")

# Dicionário fixo com regras válidas (substitui upload)
grupo_para_subgrupo_validos = {
    'CAPACIDADE': ['MÃO DE OBRA TEMPORÁRIA'],
    'CONTRATOS': ['SERVIÇOS E SOLUÇÕES', 'LICENCIAMENTO', 'MATERIAIS E CONSUMÍVEIS', 'IFRS', 'ASSINATURAS', 'SERVIÇOS DE CONSULTORIA', 'MANUTENÇÃO', 'IMPOSTOS E TAXAS', 'SATÉLITES, LINKS E TRANSMISSÃO', 'DRONE', 'CLOUDIFICATION'],
    'DESPESAS DE PRODUÇÃO': ['JORNALISMO', 'DEMAIS', 'ESTÚDIOS GLOBO', 'ESPORTE', 'CLOUDIFICATION', 'PD', 'TV GLOBO', 'E&T'],
    'OUTRAS DESPESAS': ['TREINAMENTO', 'TRANSPORTE', 'ALIMENTAÇÃO', 'DEMAIS VIAGENS', 'SERVIÇOS PATRIMONIAIS', 'DEMAIS OUTRAS DESPESAS', 'IMPRESSÕES'],
    'PROJETOS': ['PROJETOS']
}

def validar_grupo_subgrupo(grupo, subgrupo):
    return subgrupo in grupo_para_subgrupo_validos.get(grupo, [])

if historico_file and novos_file:
    historico = pd.read_excel(historico_file)
    novos = pd.read_excel(novos_file)

    colunas_entrada = ['Centro de Custo_Hyp', 'Conta', 'Centro de Resultado_Hyp', 'Projeto_Hyp', 'Finalidade']

    historico = historico.dropna(subset=colunas_entrada + ['Grupos', 'Subgrupo/Torre'])
    le_cols = {}
    for col in colunas_entrada + ['Grupos', 'Subgrupo/Torre']:
        le = LabelEncoder()
        historico[col + '_enc'] = le.fit_transform(historico[col].astype(str))
        le_cols[col] = le

    X = historico[[c + '_enc' for c in colunas_entrada]]
    y_grupo = historico['Grupos_enc']
    y_subgrupo = historico['Subgrupo/Torre_enc']

    modelo_grupo = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    scores_grupo = cross_val_score(modelo_grupo, X, y_grupo, cv=5)
    
    
    st.sidebar.markdown("### Avaliação do Modelo (Grupo)")
    st.sidebar.text(f"Acurácia média: {scores_grupo.mean():.2%}")
    st.sidebar.text(f"Desvio padrão: {scores_grupo.std():.2%}")

    modelo_subgrupo = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    scores_subgrupo = cross_val_score(modelo_subgrupo, X, y_subgrupo, cv=5)
    
    
    st.sidebar.markdown("### Avaliação do Modelo (Subgrupo)")
    st.sidebar.text(f"Acurácia média: {scores_subgrupo.mean():.2%}")
    st.sidebar.text(f"Desvio padrão: {scores_subgrupo.std():.2%}")

    y_pred_subgrupo = cross_val_predict(modelo_subgrupo, X, y_subgrupo, cv=5)
    cm = confusion_matrix(y_subgrupo, y_pred_subgrupo)
    labels_sub = le_cols['Subgrupo/Torre'].inverse_transform(np.unique(y_subgrupo))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sub)
    disp.plot(ax=ax, xticks_rotation=90, cmap='Blues', colorbar=False)
    #plt.title("Matriz de Confusão - Subgrupo")
    #st.pyplot(fig)

    modelo_grupo.fit(X, y_grupo)
    modelo_subgrupo.fit(X, y_subgrupo)

    for col in colunas_entrada:
        novos[col] = novos[col].astype(str)
        novos[col + '_enc'] = novos[col].map(
            lambda x: le_cols[col].transform([x])[0] if x in le_cols[col].classes_ else -1
        )

    X_novos = novos[[c + '_enc' for c in colunas_entrada]]
    grupo_pred = modelo_grupo.predict(X_novos)
    subgrupo_pred = modelo_subgrupo.predict(X_novos)

    novos['Grupo_Previsto'] = le_cols['Grupos'].inverse_transform(grupo_pred)
    novos['Subgrupo_Previsto'] = le_cols['Subgrupo/Torre'].inverse_transform(subgrupo_pred)

    # Corrigir subgrupos inválidos
    corrigidos = 0
    for idx, row in novos.iterrows():
        if not validar_grupo_subgrupo(row['Grupo_Previsto'], row['Subgrupo_Previsto']):
            grupo = row['Grupo_Previsto']
            candidatos = historico[historico['Grupos'] == grupo]['Subgrupo/Torre']
            if not candidatos.empty:
                mais_comum = candidatos.mode().iloc[0]
                novos.at[idx, 'Subgrupo_Previsto'] = mais_comum
                corrigidos += 1

    novos['Valido'] = novos.apply(lambda row: 'Sim' if validar_grupo_subgrupo(row['Grupo_Previsto'], row['Subgrupo_Previsto']) else 'Não', axis=1)

    # Medir nova acurácia após correções (se dados reais estiverem presentes)
    if 'Subgrupo/Torre' in novos.columns:
        try:
            y_true = novos['Subgrupo/Torre'].map(lambda x: le_cols['Subgrupo/Torre'].transform([x])[0] if x in le_cols['Subgrupo/Torre'].classes_ else -1)
            y_pred_corrigido = novos['Subgrupo_Previsto'].map(lambda x: le_cols['Subgrupo/Torre'].transform([x])[0] if x in le_cols['Subgrupo/Torre'].classes_ else -1)
            acuracia_corrigida = accuracy_score(y_true, y_pred_corrigido)
            st.sidebar.markdown("### Acurácia pós-correção")
            st.sidebar.text(f"{acuracia_corrigida:.2%} (baseado em Subgrupo/Torre original vs previsto)")
        except:
            pass

    resultado_final = novos[colunas_entrada + ['Grupo_Previsto', 'Subgrupo_Previsto', 'Valido']]
    st.success("Classificação concluída! Veja o resultado abaixo:")
    st.dataframe(resultado_final)
    

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        resultado_final.to_excel(writer, index=False)
    st.download_button("📂 Baixar Excel com Previsões", data=output.getvalue(), file_name="Novos_Mapeados.xlsx")

    if salvar_historico:
        historico_atualizado = pd.concat([
            historico[colunas_entrada + ['Grupos', 'Subgrupo/Torre']].copy(),
            resultado_final.rename(columns={
                'Grupo_Previsto': 'Grupos',
                'Subgrupo_Previsto': 'Subgrupo/Torre'
            })
        ], ignore_index=True)

        historico_output = BytesIO()
        with pd.ExcelWriter(historico_output, engine='openpyxl') as writer:
            historico_atualizado.to_excel(writer, index=False)

        st.download_button(
            label="📂 Baixar Histórico Atualizado",
            data=historico_output.getvalue(),
            file_name=f"Historico_Atualizado_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        )

else:
    st.info("🔍 Por favor, envie os dois arquivos no menu lateral para iniciar a classificação.")
