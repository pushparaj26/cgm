#mindfulness index

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Set notebook mode to work in offline
from movement_dash import preprocess

pd.options.plotting.backend = "plotly"
file_loction="m_data/"


def get_sleep_signal(df):
  sleep_hours=(22,5)
  df=df[(df.hour>sleep_hours[0])|(df.hour<sleep_hours[1])]
  return df

@st.cache(allow_output_mutation=True)
def load_data(fituserid=8378563200):
    sleep_df = pd.read_csv("fitbit_sleep/fitbit_export_sleep.csv", skiprows=1)
    sleep_df = sleep_df.dropna()
    return sleep_df

def get_score(range_val, val):
    for i in range(len(range_val)):
        if val >= range_val[i]:
            return i+1
    return 5

def get_score_reverse(range_val, val):
    for i in range(len(range_val)):
        if val <= range_val[i]:
            return i+1
    return 5


def get_score_by_segment(sleepseg_val,score_r,e,g,l):
    if sleepseg_val in score_r:
        return e
    elif sleepseg_val<score_r[0]:
        return l
    elif sleepseg_val>score_r[-1]:
        return g



# calculate sleep segment score for deep sleep
def sleep_Deep_score(TSD,score_r,sleepseg_val):
    if TSD==4:
        if sleepseg_val<=score_r[0]:
            return 1
        else:
            return 2
    elif TSD==5:
        return get_score_by_segment(sleepseg_val,score_r[1],2,3,1)
    elif TSD==6:
        return get_score_by_segment(sleepseg_val,score_r[2],3,4,2)
    elif TSD==7:
        return get_score_by_segment(sleepseg_val,score_r[3],4,5,3)
    elif TSD>=8:
        return get_score_by_segment(sleepseg_val,score_r[3],5,5,4)
    else:
        return 10


# calculate sleep segment score light and rem sleep
def sleep_LT_REM_score(TSD,score_r,sleepseg_val):
    if TSD==4:
        if sleepseg_val<=score_r[0]:
            return 1
        else:
            return 2
    elif TSD==5:
        return get_score_by_segment(sleepseg_val,score_r[1],2,3,1)
    elif TSD==6:
        return get_score_by_segment(sleepseg_val,score_r[2],3,4,2)
    elif TSD==7:
        return get_score_by_segment(sleepseg_val,score_r[3],4,5,3)
    elif TSD>=8:
        return get_score_by_segment(sleepseg_val,score_r[3],1,5,4)
    else:
        return 10



def app():
    st.markdown('''
            # **Sleep**
            ---
            ''')

    #st.sidebar.title("**Sleep Index**")

    sleep_df=load_data()

    import pandas as pd

    avg_sleep_hour = np.round(sleep_df['Minutes Asleep'].mean() / 60, 2)
    avg_REM_hour = np.round(sleep_df['Minutes REM Sleep'].mean() / 60, 2)
    avg_Deep_hour = np.round(sleep_df['Minutes Deep Sleep'].mean() / 60, 2)
    avg_light_hour = np.round(sleep_df['Minutes Light Sleep'].mean() / 60, 2)

    avg_REM_min = np.round(sleep_df['Minutes REM Sleep'].mean(), 0)
    avg_Deep_min = np.round(sleep_df['Minutes Deep Sleep'].mean(), 0)
    avg_light_min = np.round(sleep_df['Minutes Light Sleep'].mean(), 0)
    avg_sleep_min = np.round(sleep_df['Minutes Asleep'].mean(), 0)
    avg_awake_min = np.round(sleep_df['Minutes Awake'].mean(), 0)
    TSD = round(avg_sleep_hour)

    #sleep score ranges
    Deep_SL_R = [36, list(range(45, 75)), list(range(54, 90)), list(range(63, 105)), list(range(72, 120))]
    REM_SL_R = [96, list(range(120, 165)), list(range(144, 198)), list(range(168, 231)), list(range(192, 264))]
    Light_SL_R = [48, list(range(60, 75)), list(range(72, 90)), list(range(84, 105)), list(range(96, 120))]

    TSP_R = [299, 359, 419, 479]
    AWake_TM_R = [30, 20, 15, 5]




    pm1, pm2, pm3, pm4,pm5  = st.columns((1, 1, 1,1, 1 ))

    pm1.markdown('**Total Sleep**')
    pm1.subheader(f'{avg_sleep_min} Min')

    pm2.markdown('**REM Sleep**')
    pm2.subheader(f'{avg_REM_min} Min')

    pm3.markdown('**Deep Sleep**')
    pm3.subheader(f'{avg_Deep_min} Min')

    pm4.markdown('**Light Sleep**')
    pm4.subheader(f'{avg_light_min} Min')

    pm5.markdown('**Awake Time**')
    pm5.subheader(f'{avg_awake_min} Min')

    TSP_S = get_score_reverse(TSP_R, avg_sleep_min)
    Awake_time_score = get_score(AWake_TM_R, avg_awake_min)
    deep_sleep_S = sleep_Deep_score(TSD, Deep_SL_R, avg_Deep_min)
    rem_sl_S = sleep_LT_REM_score(TSD, REM_SL_R, avg_REM_min)
    light_sl_S = sleep_LT_REM_score(TSD, Light_SL_R, avg_light_min)

    st.markdown("""---""")

    sleep_index_score = (TSP_S * 0.15) + (Awake_time_score * 0.20) + (deep_sleep_S * 0.30) + (light_sl_S * 0.15) + (
                rem_sl_S * 0.20)

    slp1S, slp2S, slp3S = st.columns((1, 1, 1))
    slp1S.markdown('')
    slp2S.markdown('**Human Edge Score**')
    # CGM2S.markdown("""---""")
    slp2S.subheader(f'        {sleep_index_score}   ')
    slp3S.markdown('')



    import pandas as pd
    df = pd.DataFrame(dict(
        score=[rem_sl_S, TSP_S, deep_sleep_S, Awake_time_score,light_sl_S],
        metric=['REM Sleep', 'Total Sleep Duration', 'Deep Sleep', 'Awake','Light Sleep'
                ]))


    st.markdown("""---""")
    Redarfig = px.line_polar(df, r='score', theta='metric', line_close=True)
    st.subheader("Metrices Score :")
    st.plotly_chart(Redarfig, use_container_width=True)


    st.markdown("""---""")

    sleep_df['Start Time'] = pd.to_datetime(sleep_df['Start Time'])
    sleep_df['End Time'] = pd.to_datetime(sleep_df['End Time'])

    sleep_df['pre_date'] = sleep_df['End Time'] - pd.to_timedelta(1, unit='d')
    sleep_df['Date'] = sleep_df['pre_date'].dt.date
    #sleep_df['Date'] = sleep_df['Date'].astype(str)

    ycols = ["Minutes REM Sleep", "Minutes Light Sleep", "Minutes Deep Sleep"]
    sleep_qualityfig = px.bar(sleep_df, x="Date", y=ycols, title="Sleep Time line", text_auto=True)
    sleep_qualityfig.update_layout(title_x=0,
                                   margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Date",
                                   yaxis_title="Sleep Minutes")

    st.plotly_chart(sleep_qualityfig)

    st.markdown("""---""")

    mean_sleep = sleep_df[["Minutes REM Sleep", "Minutes Light Sleep", "Minutes Deep Sleep"]].mean().reset_index(
        name='Total')
    mean_sleep.columns = ['Type', 'Sleep Minutes']

    sleep_distribution = px.pie(mean_sleep, values='Sleep Minutes', names='Type')
    sleep_distribution.update_layout(title_text="Sleep Distribution")

    st.plotly_chart(sleep_distribution)













