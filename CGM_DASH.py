import pandas as pd
import plotly.express as px
import streamlit as st
import altair as alt
import cgmquantify as cgm
import streamlit_tags as st


names = ['Marcus','Manish']
usernames = ['Marcus','Manish']
passwords = ['Marcus@cgm','Manish@cgm']

pd.options.plotting.backend = "plotly"




st.set_page_config(layout='wide', initial_sidebar_state='expanded')



def calc_slope(x):
    try:
        slope = np.polyfit(range(len(x)), x, 1)[0]
    except:
        return None
    return slope

import numpy as np
def calc_slope(x):
    try:
        slope = np.polyfit(range(len(x)), x, 1)[0]
    except:
        return None
    return slope

def cal_culate_peak_stats(slope):

    if slope.shape[0]==1:
        return {}
    slope=slope.sort_values('Actual time')
    max_bg=slope['Glucose reading'].max()
    peak_time=slope[slope['Glucose reading']==max_bg]['Actual time'].max()
    upslope=slope[slope['Actual time']<=peak_time]
    down_slope=slope[slope['Actual time']>peak_time]
    stats={}
    stats['time_to_reach_top']=(upslope['Actual time'].max()-upslope['Actual time'].min()).total_seconds() / 60.0
    stats['time_to_come_down']=(down_slope['Actual time'].max()-down_slope['Actual time'].min()).total_seconds() / 60.0
    stats['start_time']=upslope['Actual time'].min()
    stats['end_time']=upslope['Actual time'].max()
    stats['max_BG']=max_bg
    stats['Total_sample']=slope.shape[0]
    stats['upslope_sample']=upslope.shape[0]
    stats['up_start_BG']=upslope['Glucose reading'].min()
    stats['down_end_BG']=down_slope['Glucose reading'].min()
    stats['downslope_sample']=down_slope.shape[0]
    stats['peak_time']=peak_time
    return stats


def preprocess_data(cgm_df):
    format = '%d-%m-%Y %H:%M%p'
    try:
        cgm_df['Actual time'] = pd.to_datetime(cgm_df['Actual time'], format=format)
    except:
        cgm_df['Actual time'] = pd.to_datetime(cgm_df['Actual time'])


    cgm_df = cgm_df[['Actual time', 'Glucose reading']]
    df = cgm_df[cgm_df['Glucose reading'].notnull()]
    df = df.sort_values('Actual time')
    df['diff_next_sample'] = df['Actual time'].diff().dt.days
    df['tag_no'] = df['diff_next_sample'] > 1
    df['tag_no'] = df['tag_no'].cumsum()
    #df['month'] = df['Actual time'].dt.to_period('M').astype(str)
    df['Tag No'] = df.tag_no.apply(lambda x: f'Tag_No - {x+1}')


    return df


@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')


def perform_analysis(glucose):

    glucose = glucose.sort_values('Actual time')
    glucose = glucose[['Actual time', 'Glucose reading']]
    import pandas as pd
    glucose['Actual time'] = pd.to_datetime(glucose['Actual time'])
    glucose['Day']=glucose['Actual time'].dt.date
    glucose=glucose.reset_index(drop=True)


    #Add Start end date to dashboard
    dt1,  dt2,  dt3 = st.columns((1, 1, 1))

    dt1.markdown("**FROM**")#('FROM')
    dt1.write(str(glucose["Actual time"].min().date()))



    dt2.markdown('**TO**')
    dt2.write(str(glucose["Actual time"].max().date()))

    dt3.markdown('**TOTAL DAYS**')
    dt3.write(str((glucose["Actual time"].max() - glucose["Actual time"].min()).days))

   # Glucose Matrices calculation
   # st.dataframe(data=glucose, width=1024, height=768)
    
    grop1 = len(glucose[glucose['Glucose reading'].between(161,250)])
    grop2 = len(glucose[glucose['Glucose reading'].between(131,160)])
    grop3 = len(glucose[glucose['Glucose reading'].between(90,130)])
    grop4 = len(glucose[glucose['Glucose reading'].between(75,89)])
    grop5 = len(glucose[glucose['Glucose reading'].between(40,69)])
    

    
    comp_glucose =glucose['Glucose reading']
    YY = []
    for k in range(len(comp_glucose)):
            if np.all(comp_glucose[k]>160):
                  YY.append('very_high')
            elif np.all(130<comp_glucose[k]<=160):
                 YY.append('high')
            elif np.all(90<comp_glucose[k]<=130):
                YY.append('Normal')
            elif np.all(75<comp_glucose[k]<=90):
                YY.append('low')
            else:
                YY.append('very_low')
    glucose['group_status'] =YY      
    #import plotly.graph_objects as go
    #st.dataframe(data=glucose, width=1024, height=768)
    #time_spent _ingroup=
 
    bar_rounded = alt.Chart(glucose).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, 
        ).encode(
            x='day(Day):O',
            y='count():Q',
            color='group_status:N',
            ).properties(width=50)
            
    st.altair_chart(bar_rounded, use_container_width=True )










    # m1.metric(label='FROM', value='',delta=str(glucose["Actual time"].min().date()))
    GMI_HbA1c=np.round(cgm.GMI(glucose),3)
    ADA_HbA1c=np.round(cgm.eA1c(glucose),3)
    pTIR=np.round(cgm.PInsideRange(glucose), 3)
    Adrr=np.round(cgm.ADRR(glucose), 3)



    glucose['upper'] = glucose['Glucose reading'] > 110
    glucose['previuos'] = glucose['upper'].shift()
    glucose['next'] = glucose['upper'].shift(-1)
    glucose[['upper', 'previuos', 'next']] = glucose[['upper', 'previuos', 'next']].fillna(False)
    glucose['isupper'] = glucose.apply(lambda x: (x['upper'] | x['previuos'] | x['next']), axis=1)

    upper = glucose[glucose.isupper]


    upper['is_new_segment'] = upper.reset_index()['index'].diff().values > 1
    upper['peak'] = upper['is_new_segment'].cumsum()

    max_BG_peak = upper.groupby('peak')['Glucose reading'].max().reset_index()

    peak_day = upper.groupby(upper['Actual time'].dt.date)['peak'].nunique().reset_index()
    peak_day.columns = ['Date', 'Total Peaks']

    time_spent_in_peak = upper.groupby('peak')['Actual time'].apply(
        lambda x: (max(x) - min(x)).total_seconds() / 60.0) + 1
    time_spent_in_peak = time_spent_in_peak.reset_index()

    time_spent_in_peak.columns = ['Peak No', 'Time in spike (Min)']


    # spikes average calculation

    

    AvgMax_in_peak=np.round(np.mean(max_BG_peak["Glucose reading"]),2)
    Average_peaks_eachday=np.round(np.mean(peak_day["Total Peaks"]),2)
    average_time_spent_in_peaks=np.round(np.mean(time_spent_in_peak["Time in spike (Min)"]),2)

    average_bg=np.round(np.mean(glucose[ 'Glucose reading']),2)

    



    # spikes Stats calculation

    max_BG_peak.columns = ['Spike No', 'Max Glucose']
    peak_day.columns = ['Date', 'Spikes']
    time_spent_in_peak.columns = ['Spike No', 'Time in spike (Min)']

    stats = upper.groupby('peak').apply(cal_culate_peak_stats)
    import pandas as pd
    stats_df = pd.DataFrame.from_dict(list(stats.values))
    stats_df['peak'] = stats.index
    stats_df = stats_df.rename(columns={'peak': 'Spike No'})
    stats_df = stats_df[stats_df.upslope_sample != 1]
    stats_df['Rising_Rate'] = stats_df.apply(lambda x: (x['max_BG'] - x['up_start_BG']) / x['time_to_reach_top'],
                                             axis=1)

    stats_df['Falling_Rate'] = stats_df.apply(
        lambda x: (x['max_BG'] - x['down_end_BG']) / x['time_to_come_down'] if x['time_to_come_down'] != 0 else 0,
        axis=1)
    import pandas as pd
    stats_df = pd.merge(stats_df, time_spent_in_peak, on='Spike No', how='left')
    stats_df = pd.merge(stats_df, max_BG_peak, on='Spike No', how='left')
    stats_df = stats_df.rename(columns={'peak_time': 'Spike Time'})

    cols = ['Spike No', 'time_to_reach_top', 'time_to_come_down', 'Rising_Rate', 'Falling_Rate', 'Time in spike (Min)',
            'Max Glucose', 'start_time', 'end_time', 'max_BG', 'Total_sample', 'upslope_sample', 'up_start_BG',
            'down_end_BG'
        , 'downslope_sample', 'Spike Time']
    stats_df = stats_df[cols]








    ##### Rader Chart
    ATtoFloorfrom_Peak = np.round(stats_df[stats_df.time_to_come_down > 0]['time_to_come_down'].mean(), 2)
    ATtoTopfrom_base = np.round(stats_df[stats_df.time_to_come_down > 0]['time_to_reach_top'].mean(), 2)

    ## Average time in spike add to DB
    
   

    Average_peaks_eachdayR = [6, 5, 4, 3]
    ATtoFloorfrom_PeakR = [151, 91, 60, 31]
    average_time_spent_in_peaksR = [67, 51, 18, 19, 3]
    ADRRR = [50, 40, 30, 19, 10]
    GMI_HbA1cR = [8, 6.5, 5.7, 5]

    AvgMax_in_peakR=[181,180,150,140]
    average_bgR=[180,140,130,110]




    def get_score(range_val, val):
        for i in range(len(range_val)):
            if val >= range_val[i]:
                return i+1
        return 5

    TINRR = [40, 60, 75, 95]
    #ATtoTopfrom_baseR = [46, 47, 48, 49, 50]


    def get_score_reverse(range_val, val):
        for i in range(len(range_val)):
            if val <= range_val[i]:
                return i+1
        return 5

    AvgMax_in_peak_Score=get_score(AvgMax_in_peakR, AvgMax_in_peak)
    Average_peaks_eachday_Score = get_score(Average_peaks_eachdayR, Average_peaks_eachday)
    ATtoFloorfrom_PeakR_score = get_score(ATtoFloorfrom_PeakR, ATtoFloorfrom_Peak)
    average_time_spent_in_peaks_score = get_score(average_time_spent_in_peaksR, average_time_spent_in_peaks)

    average_bg_Score=get_score(average_bgR,average_bg)

    ADRR_score = get_score(ADRRR, Adrr)
    TINR_score = get_score_reverse(TINRR, pTIR)
    #ATtoTopfrom_base_score = get_score_reverse(ATtoTopfrom_baseR, ATtoTopfrom_base)
    GMIA1c_Score = get_score(GMI_HbA1cR, GMI_HbA1c)

    CGM_index_Score = (0.1 * GMIA1c_Score) + (0.15 * Average_peaks_eachday_Score) + (
                0.15 * average_time_spent_in_peaks_score) + (0.1 * ATtoFloorfrom_PeakR_score) + (
                                  0.2 * AvgMax_in_peak_Score) + (0.2 * TINR_score) + (0.1 * average_bg_Score)

    CGM_index_Score=np.round(CGM_index_Score,3)

    

    #html export
      
    
    
    
    

    tabh,tabz =st.tabs(["HE SCORE", "glucose range"])
    with tabh:
         st.metric("Human Edge Score", CGM_index_Score )
         #st.markdown('**Human Edge Score**')
         #st.subheader(f'    {CGM_index_Score} ')
         #col1.metric("Temperature", "70 °F", "1.2 °F")
    with tabz:

         Labels= ['Very High', 'High', 'In Range', 'Low', 'Very Low']
         sizes= [grop1, grop2, grop3, grop4, grop5]
         #explode = (0.4, 0, 0, 0, 0.4)
         
         totalgrop = grop1 + grop2 + grop3 + grop4 + grop5
         Pgrop1= round(grop1/totalgrop*100,2)
         Pgrop2= round( grop2/totalgrop*100,2)
         Pgrop3= round( grop3/totalgrop*100,2)
         Pgrop4= round( grop4/totalgrop*100,2)
         Pgrop5= round( grop5/totalgrop*100,2)
         
         pg1, pg2, pg3, pg4, pg5 = st.columns(5)
          
         pg1.metric("Very high(%)", value=Pgrop1)
         pg2.metric("High(%)", Pgrop2)
         pg3.metric("Normal(%)", Pgrop3 )
         pg4.metric("Low(%)", Pgrop4 )
         pg5.metric("Very Low(%)", Pgrop5 )
         
         figz= px.pie(sizes, values=sizes, names = Labels, title = 'Pie chart for time range')
         figz.update_traces(textposition= 'inside', textinfo= 'percent+label')
         figz.update_layout(title_font_size = 42 )
         st.plotly_chart(figz)
         

              

         #-------------------------INTERACTIVE CHART--------------
        # interactive glucose reading analysis
         base = alt.Chart(glucose).properties(width=900)
         selection = alt.selection_multi(fields=['group_status'], bind='legend')
         line = base.mark_line().encode(
             x='Actual time',
             y='Glucose reading',
             color='group_status:N',
             strokeDash='group_status',
             opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
         ).add_selection(selection)

         rule = base.mark_rule().encode(
             y='average(Glucose reading)',
             color='group_status',
             size=alt.value(2)
         )

         chart = line 

         
         selection1 = alt.selection_multi(fields=['group_status'])
         chart2 = alt.Chart(glucose).mark_bar().encode(
            x='count()',
            y='group_status:N'
          ).add_selection(selection1)
          

          
         chart4 = alt.vconcat(chart, chart2, data = glucose, title = "glucose reading analysis" )
         st.altair_chart(chart4, theme= None, use_container_width= True)


  
    
         
         
    
    
    #tabs header
    
    tab1, tab2, tab3 = st.tabs(["HbA1", "Spike", "HE Score"])
    with tab1:
         st.subheader("Glucose Matrices calculation")
         gm2, gm3, gm4  = st.columns((1, 1, 1))
         gm2.metric("American Diabetes Association-HbA1c", ADA_HbA1c )
         gm3.metric("Average Daily Risk Range", Adrr )
         gm4.metric("Percent time inside range 70-150", pTIR )
         
         #gm1.markdown('**Glucose management index-HbA1c**')
         #gm1.subheader(GMI_HbA1c)
         #gm2.markdown('**American Diabetes Association-HbA1c**')
         #gm2.subheader(ADA_HbA1c)
         #gm3.markdown('**Average Daily Risk Range**')
         #gm3.subheader(Adrr)
         #gm4.markdown('**Percent time inside range 70-150**')
         #gm4.subheader(pTIR)
    with tab2:
        st.header("spikes average calculation")
        pm1, pm2, pm3 = st.columns((1, 1, 1))

        pm1.markdown('**Daily Average spike (>110)**')
        pm1.subheader(Average_peaks_eachday)

        pm2.markdown('**Average Max BG in spike**')
        pm2.subheader(AvgMax_in_peak)
        pm3.markdown('**Average Time in spike**')
        pm3.subheader(f'{average_time_spent_in_peaks} Min')
        
        st.markdown("""---""")
        with st.expander(" **Time Spent in Spike**"):
            time_spent_in_peak = pd.merge(time_spent_in_peak, stats_df[['Spike No', 'Spike Time']], on='Spike No', how='inner')
            time_spent_in_peak = time_spent_in_peak[time_spent_in_peak['Time in spike (Min)'] < 600]
            fig3 = px.line(time_spent_in_peak, x="Spike No", y='Time in spike (Min)', hover_data={'Spike Time': True})
            fig3.update_traces(marker_color='#d40707')
            fig3.update_layout(title_text="Time Spent in Spike", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
            st.plotly_chart(fig3)

        st.markdown("""---""")

        with st.expander(" **Spikes in a Day**"):
            fig1 = px.bar(peak_day, x="Date", y='Spikes')
            fig1.update_traces(marker_color='#191970')
            fig1.update_layout(title_text="Spikes in a Day", title_x=0, margin=dict(l=5, r=15, b=15, t=50))
            st.plotly_chart(fig1)
            st.write('''
            The chart above shows some numbers.
            It rolled actual dice for these, so they're guaranteed to  be random.''')
    with tab3:
        st.header("hes")
        ATTF, ATTTOP,AverageBG_DB = st.columns((1, 1, 1))

        ATTF.markdown('**Average time to Peak (Min)**')
        ATTF.subheader(ATtoFloorfrom_Peak)

        ATTTOP.markdown('**Average time to floor (Min)**')
        ATTTOP.subheader(ATtoTopfrom_base)

        AverageBG_DB.markdown('**Average BG**')
        AverageBG_DB.subheader(average_bg)
    
    
    
    
    
    
    
    

    import pandas as pd
    df = pd.DataFrame(dict(
        score=[Average_peaks_eachday_Score, ADRR_score, TINR_score, average_time_spent_in_peaks_score
             , ATtoFloorfrom_PeakR_score, AvgMax_in_peak_Score,average_bg_Score],
        metric=['Daily Average spike (>110)', 'Average Daily Risk Range',
                'Percent time inside range 70-150', 'Average Time in spike', 'Average time to Peak',
                'Average Maximum in each peak','Average Blood Glucose'
                ]))





    #Redarfig = px.scatter(df, y='score', x='metric',  size_max=60)
    Redarfig = px.line_polar(df, r='score', theta='metric', line_close=True)
    st.markdown("""---""")

    st.subheader("Metrices Score :")
    #st.plotly_chart(Redarfig, use_container_width=True,theme="streamlit")
    st.plotly_chart(Redarfig, use_container_width=True)
    # Line chart ploting on dash board

    st.markdown("""---""")

    genred = st.sidebar.selectbox("Select Patient group",('Normal person without diabetes','Official ADA recommendation for a person with diabetes') )
    
    if genred=='Official ADA recommendation for a person with diabetes':
       glucose['HE 130 mg/dl']= 130
       glucose['HE 80 mg/dl'] = 80
       fig_GR = px.line(glucose, x="Actual time", y=["Glucose reading", 'HE 130 mg/dl', 'HE 80 mg/dl'])
       
    else :
       glucose['HE 99 mg/dl']= 99
       glucose['HE 70 mg/dl'] = 70
       fig_GR = px.line(glucose, x="Actual time", y=["Glucose reading", 'HE 99 mg/dl', 'HE 70 mg/dl'])
       

    fig_GR.update_traces(marker_color='#006400')
    #fig_GR.add_trace(go.Scatter(x="Actual time", y=["Glucose reading", 'HE 150', 'HE 70'],line_color='rgb(0,100,80)'))
    fig_GR.update_layout(title_text="Glucose Reading", title_x=0, margin=dict(l=0, r=15, b=15, t=50),
                         yaxis_title='Glucose mg/dl', xaxis_title='datetime')

    st.plotly_chart(fig_GR, use_container_width=True)
    with st.expander(" **Glucose Reading explanation**"):
        st.text(''' If you get sick, your blood sugar can be hard to manage. You may not be able to eat or drink as much as usual, 
which can affect blood sugar levels. If you're ill and your blood sugar is 240 mg/dL or above, use an over-the-counter ketone test kit to 
check your urine for ketones and call your doctor if your ketones are high. High ketones can be an early sign of diabetic ketoacidosis, 
which is a medical emergency and needs to be treated immediately''')
        
        
    #st.info(f'Peaks Max BG Average : **{}**')
    st.markdown("""---""")
    with st.expander(" **Maximum Glucose in spike**"):
        max_BG_peak = pd.merge(max_BG_peak, stats_df[['Spike No', 'Spike Time']], on='Spike No', how='inner')
    
        fig2 = px.line(max_BG_peak, x="Spike No", y="Max Glucose",hover_data={'Spike Time':True})
        fig2.update_traces(marker_color='#d40707')
        fig2.update_layout(title_text="Maximum Glucose in spike", title_x=0, margin=dict(l=0, r=15, b=15, t=50))
        st.plotly_chart(fig2)

    

    

    #fig=upper.groupby(upper['Actual time'].dt.date)['peak'].nunique().plot()
    st.markdown("""---""")
    #st.markdown("Glucose_readings throughout a day")
    #st.image("gcm_day.png", width=800)
    #fig = px.line(glucose,x="Actual time", y="Glucose reading", title="Blood Glucose Timeline")
    #st.plotly_chart(fig)
    



    st.download_button(
    "Press to Download table",
    convert_df(stats_df),
    "cgm_stats.csv",
    "text/csv",
    key='download-csv')
    #st.dataframe(data=stats_df, width=1024, height=768)
    
    
    #PDFbyte = doc1.read()



    #with open("cgm_rep.pdf", "rb") as pdf_file:
            #PDFbyte = doc1.read()
   # if doc1:
        #st.download_button(label="Export Report Pdf",
           #         data=bio.getvalue(),
          #  file_name="Report.docx",
           # mime="docx")
    
   
    
    
    
    

    
    

def open_dash(df):
    cgmdf = preprocess_data(df)

    cgmdf['date'] = pd.to_datetime(cgmdf['Actual time'].dt.date)
    tags_dates = cgmdf.groupby('Tag No')['date'].agg({'min', max}).reset_index()

    tags_dates.columns = ['Tag No', 'start date', 'end date']
    #st.sidebar.dataframe(data=tags_dates)
    tags = list(cgmdf['Tag No'].unique())

    tagnum = st.sidebar.selectbox("Select tag Number:", tags)

    glucose = cgmdf[cgmdf['Tag No'] == tagnum]

    daterange = st.sidebar.date_input("Select Tag Date Range",
                                      (glucose['date'].min(), glucose['date'].max()), min_value=glucose['date'].min(),
                                      max_value=glucose['date'].max())
    # st.write(type(daterange[0]))
    # st.write(type(glucose['date'].dtypes))
    if len(daterange) == 2:
        glucose = glucose[
            (glucose['date'] >= pd.to_datetime(daterange[0])) & (glucose['date'] <= pd.to_datetime(daterange[1]))]
        perform_analysis(glucose)





def app():

    with st.sidebar.header('Upload your CGM data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        unit_button = st.sidebar.selectbox("Select units:", ('mg/dl','mmol/L'))

    @st.cache(allow_output_mutation=True)
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv

    st.sidebar.title("**Metabolic**")


    # Web App Title
    st.title(':purple[Continuous Glucose Monitoring]')

    if uploaded_file is not None:
        df = load_csv()  # pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
        open_dash(df)
    else:
        local_df = pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
        open_dash(local_df)







