import pandas as pd
import plotly.express as px
import streamlit as st

import cgmquantify as cgm

names = ['Marcus','Manish']
usernames = ['Marcus','Manish']
passwords = ['Marcus@cgm','Manish@cgm']

pd.options.plotting.backend = "plotly"



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
    glucose['HE 150']=150
    glucose['HE 70'] = 70


   # Glucose Matrices calculation

    st.markdown("""---""")

    gm1, gm2, gm3 ,gm4 = st.columns((1, 1, 1,1))

    # m1.metric(label='FROM', value='',delta=str(glucose["Actual time"].min().date()))
    GMI_HbA1c=np.round(cgm.GMI(glucose),3)
    ADA_HbA1c=np.round(cgm.eA1c(glucose),3)
    pTIR=np.round(cgm.PInsideRange(glucose), 3)
    Adrr=np.round(cgm.ADRR(glucose), 3)





    gm1.markdown('**Glucose management index-HbA1c**')
    gm1.subheader(GMI_HbA1c)
    gm2.markdown('**American Diabetes Association-HbA1c**')
    gm2.subheader(ADA_HbA1c)
    gm3.markdown('**Average Daily Risk Range**')
    gm3.subheader(Adrr)
    gm4.markdown('**Percent time inside range 70-150**')
    gm4.subheader(pTIR)



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

    st.markdown("""---""")

    AvgMax_in_peak=np.round(np.mean(max_BG_peak["Glucose reading"]),2)
    Average_peaks_eachday=np.round(np.mean(peak_day["Total Peaks"]),2)
    average_time_spent_in_peaks=np.round(np.mean(time_spent_in_peak["Time in spike (Min)"]),2)

    average_bg=np.round(np.mean(glucose[ 'Glucose reading']),2)

    pm1, pm2, pm3 = st.columns((1, 1, 1))

    pm1.markdown('**Daily Average spike (>110)**')
    pm1.subheader(Average_peaks_eachday)

    pm2.markdown('**Average Max BG in spike**')
    pm2.subheader(AvgMax_in_peak)
    pm3.markdown('**Average Time in spike**')
    pm3.subheader(f'{average_time_spent_in_peaks} Min')



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
    st.markdown("""---""")
    ATTF, ATTTOP,AverageBG_DB = st.columns((1, 1,1))

    ATTF.markdown('**Average time to Peak (Min)**')
    ATTF.subheader(ATtoFloorfrom_Peak)

    ATTTOP.markdown('**Average time to floor (Min)**')
    ATTTOP.subheader(ATtoTopfrom_base)

    AverageBG_DB.markdown('**Average BG**')
    AverageBG_DB.subheader(average_bg)

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

    st.markdown("""---""")
    CGM1S, CGM2S, CGM3S = st.columns((1, 1, 1))
    CGM1S.markdown('')
    CGM2S.markdown('**Human Edge Score**')
    #CGM2S.markdown("""---""")
    CGM2S.subheader(f'    {CGM_index_Score} ')
    CGM3S.markdown('')

    import pandas as pd
    df = pd.DataFrame(dict(
        score=[Average_peaks_eachday_Score, GMIA1c_Score, ADRR_score, TINR_score, average_time_spent_in_peaks_score
            , ATtoFloorfrom_PeakR_score, AvgMax_in_peak_Score,average_bg_Score],
        metric=['Daily Average spike (>110)', 'Glucose management index-HbA1c', 'Average Daily Risk Range',
                'Percent time inside range 70-150', 'Average Time in spike', 'Average time to Peak',
                'Average Maximum in each peak','Average Blood Glucose'
                ]))



    Redarfig = px.line_polar(df, r='score', theta='metric', line_close=True)
    st.markdown("""---""")

    st.subheader("Metrices Score :")
    st.plotly_chart(Redarfig, use_container_width=True)

    # Line chart ploting on dash board

    st.markdown("""---""")

    fig_GR = px.line(glucose, x="Actual time", y=["Glucose reading", 'HE 150', 'HE 70'])
    fig_GR.update_traces(marker_color='#264653')

    fig_GR.update_layout(title_text="Glucose Reading", title_x=0, margin=dict(l=0, r=10, b=10, t=30),
                         yaxis_title='Glucose', xaxis_title='datetime')

    st.plotly_chart(fig_GR, use_container_width=True)

    #st.info(f'Peaks Max BG Average : **{}**')
    st.markdown("""---""")

    max_BG_peak = pd.merge(max_BG_peak, stats_df[['Spike No', 'Spike Time']], on='Spike No', how='inner')

    fig2 = px.line(max_BG_peak, x="Spike No", y="Max Glucose",hover_data={'Spike Time':True})
    fig2.update_traces(marker_color='#264653')
    fig2.update_layout(title_text="Maximum Glucose in spike", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
    st.plotly_chart(fig2)

    st.markdown("""---""")
    time_spent_in_peak = pd.merge(time_spent_in_peak, stats_df[['Spike No', 'Spike Time']], on='Spike No', how='inner')
    time_spent_in_peak = time_spent_in_peak[time_spent_in_peak['Time in spike (Min)'] < 600]
    fig3 = px.line(time_spent_in_peak, x="Spike No", y='Time in spike (Min)', hover_data={'Spike Time': True})
    fig3.update_traces(marker_color='#264653')
    fig3.update_layout(title_text="Time Spent in Spike", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
    st.plotly_chart(fig3)

    st.markdown("""---""")


    fig1 = px.line(peak_day, x="Date", y='Spikes')
    fig1.update_traces(marker_color='#264653')
    fig1.update_layout(title_text="Spikes in a Day", title_x=0, margin=dict(l=0, r=10, b=10, t=30))
    st.plotly_chart(fig1)


    st.markdown("""---""")

    st.download_button(
        "Press to Download table",
        convert_df(stats_df),
        "cgm_stats.csv",
        "text/csv",
        key='download-csv'
    )

    st.dataframe(data=stats_df, width=1024, height=768)

    #fig=upper.groupby(upper['Actual time'].dt.date)['peak'].nunique().plot()


    #fig = px.line(glucose,x="Actual time", y="Glucose reading", title="Blood Glucose Timeline")
    #st.plotly_chart(fig)


def open_dash(df):
    cgmdf = preprocess_data(df)

    cgmdf['date'] = pd.to_datetime(cgmdf['Actual time'].dt.date)
    tags_dates = cgmdf.groupby('Tag No')['date'].agg({'min', max}).reset_index()

    tags_dates.columns = ['Tag No', 'start date', 'end date']
    # st.sidebar.dataframe(data=tags_dates)
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

    with st.sidebar.header('1. Upload your CGM data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    @st.cache(allow_output_mutation=True)
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv

    st.sidebar.title("**Metabolic**")


    # Web App Title
    st.markdown('''
    # **Continuous Glucose Monitoring**
    ---
    ''')

    if uploaded_file is not None:
        df = load_csv()  # pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
        open_dash(df)
    else:
        local_df = pd.read_csv('Ultrahuman_Cyborg_Manish Jain.csv')
        open_dash(local_df)







