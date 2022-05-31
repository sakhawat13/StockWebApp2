import pickle


# In[2]:


import pandas as pd 
import datetime
import yfinance as yf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from st_aggrid.shared import JsCode


# In[3]:


filename = 'classifier_model.sav'
clf = pickle.load(open(filename, 'rb'))


# In[4]:

st.title("Stock Prediction")


today = datetime.date.today()
lastfive = today - datetime.timedelta(days=23)

day = today.strftime ("%d/%m/%Y")
five = lastfive.strftime ("%d/%m/%Y")


# In[5]:


import investpy


# In[6]:


stock_df = investpy.get_stocks_overview(country="Bangladesh", 
                        as_json=False, 
                        n_results=1000)


# In[7]:

#st = list[[]]

option = list(( stock_df["name"]).unique())

opt = st.multiselect(
     'Which companies would you like?(can chose multiple)',
     (option))
all_options = st.checkbox("Select all options")

if all_options:
    opt = option

st.write('You selected:', opt)

for index, item in enumerate(opt):
    opt[index] = stock_df.loc[stock_df["name"]==item]["symbol"].values[0]

 
st.write(opt)
num_day = st.number_input('Number of days',5)



submit = st.button("Submit")

if submit:
  merged = pd.DataFrame()
  
  for s in opt:
      df4 = investpy.get_stock_historical_data(stock= s,
                                        country='Bangladesh',
                                        from_date="01/01/2007",
                                        to_date= day)
      df4["VolAvgNDays"] = df4["Volume"].rolling(15).mean()
      df4 = df4[::-1]
      df4["LP"] = df4["Close"].shift(-1)
      df4["Change"] = ((df4["Close"]-df4["LP"])/df4["LP"])*100
      df4 = df4[df4['VolAvgNDays'].notna()]
      pred1 = clf.predict(df4[["Close","Volume","VolAvgNDays","Change"]])
      df4["pred"] = pred1
      df4["Name"] = s
      
      if df4.shape[0] < num_day:
            st.write("Sorry "+ str(num_day) + " days of data for this company isnt available")
      else:
          df4 = df4[::-1]
          df4['pattern'] = df4.groupby((df4.pred != df4.pred.shift()).cumsum()).cumcount()+1
          df4 = df4[::-1]
          df5 = df4.head(num_day)
          df5 = df5[["Name","pred","pattern","Open","High","Low","Close","Volume","Change","VolAvgNDays"]]
          df5.reset_index(inplace=True)
          merged = pd.concat([merged, df5], axis=0)
          def aggrid_interactive_table(df: pd.DataFrame):
            """Creates an st-aggrid interactive table based on a dataframe.
            Args:
                df (pd.DataFrame]): Source dataframe
            Returns:
                dict: The selected row
            """
            options = GridOptionsBuilder.from_dataframe(
                df, enableRowGroup=True, enableValue=True, enablePivot=True
            )
            jscode = JsCode("""
                        function(params) {
                            if (params.data.pred === 1) {
                                return {
                                    'color': 'white',
                                    'backgroundColor': 'green'
                                }
                            }
                            if (params.data.pred === -1) {
                                return {
                                    'color': 'white',
                                    'backgroundColor': 'red'
                                }
                            }
                        };
                        """)  
            gridOptions=options.build()
            gridOptions['getRowStyle'] = jscode
            options.configure_side_bar()
            #options.configure_selection("single")
            
            selection = AgGrid(
                df,
                enable_enterprise_modules=True,
                gridOptions=gridOptions,
                
                            
                theme="dark",
                #update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True,
            )
            return selection
  selection = aggrid_interactive_table(df=merged)
#if selection:
#st.write("You selected:")
#st.json(selection["selected_rows"])

