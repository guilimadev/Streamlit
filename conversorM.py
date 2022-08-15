from ast import Return
from cgitb import text
import csv
import streamlit as st
import pandas as pd
import numpy as np 
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
 

st.title('Conversor Mapfre')
df1 = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])



def converter(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data




df_xlsx = converter(df1)
st.download_button(label="Baixar Arquivo",  data=df_xlsx, file_name="output.xlsx")


    
    