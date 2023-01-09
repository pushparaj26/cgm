import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
# Set notebook mode to work in offline
import seaborn as sns
import streamlit as st

pd.options.plotting.backend = "plotly"
file_loction="A_data/apple_health_export"



def plot_boxplot_remove_minor(df,grp,val,ratio):
  sample=df[grp].value_counts(normalize=True)#.plot.bar();
  sample=sample[sample>ratio].index
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
  df[grp].value_counts(normalize=True).plot.bar(ax=axes[0],title=f'{val} sample each {grp}');
  if ratio>0:
    sns.boxplot(x=grp, y=val, data=df[df[grp].isin(sample)],palette="Set3",ax=axes[1]).set_title(f'{val} distribution by {grp}');
  else:
    sns.boxplot(x=grp, y=val, data=df,palette="Set3",ax=axes[1]).set_title(f'{val} distribution by {grp}');


def plot_barchart(data,ptitle='steps by hour'):
  data.sort_values(ascending=False).plot.bar(figsize=(10,6),rot=45,title=ptitle);
  plt.show()


def plot_boxplot_remove_minor(df,grp,val,ratio):
  sample=df[grp].value_counts(normalize=True)#.plot.bar();
  sample=sample[sample>ratio].index
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
  df[grp].value_counts(normalize=True).plot.bar(ax=axes[0],title=f'{val} sample each {grp}');
  if ratio>0:
    sns.boxplot(x=grp, y=val, data=df[df[grp].isin(sample)],palette="Set3",ax=axes[1]).set_title(f'{val} distribution by {grp}');
  else:
    sns.boxplot(x=grp, y=val, data=df,palette="Set3",ax=axes[1]).set_title(f'{val} distribution by {grp}');


def preprocess(df):
    #df.columns=[c[1:] for c in df.columns]
    df.type=df.type.str.replace('HKQuantityTypeIdentifier','')
    df.type=df.type.str.replace('HKCategoryTypeIdentifier','')
    format = '%Y-%m-%d %H:%M:%S %z'
    df['creationDate'] = pd.to_datetime(df['creationDate'],
                                      format=format)
    df['startDate'] = pd.to_datetime(df['startDate'],
                                    format=format)
    df['endDate'] = pd.to_datetime(df['endDate'],
                                  format=format)

    df['year'] = df['startDate'].dt.year
    df['month'] = df['startDate'].dt.to_period('M')
    df['date'] = df['startDate'].dt.date
    df['day'] = df['startDate'].dt.day
    df['hour'] = df['startDate'].dt.hour
    df['dow'] = df['startDate'].dt.weekday
    df['weekday'] = df['startDate'].dt.day_name()
    df['value']=df['value'].astype(float)
    val_col=df.type.values[0]
    df[val_col]=df.value
    df=df[df.sourceName==df.sourceName.value_counts().index[0]]

    return df


@st.cache(allow_output_mutation=True)
def load_data():
    #HeartRate = pd.read_csv(f'{file_loction}/HeartRate.csv')
    StepCount = pd.read_csv(f'{file_loction}/StepCount.csv')

    #HeartRate = preprocess(HeartRate)
    StepCount = preprocess(StepCount)
    steps_by_date = StepCount.groupby(['date'])['value'].sum().reset_index(name='Steps')

    RestingHeartRate = pd.read_csv(f'{file_loction}/RestingHeartRate.csv')
    RestingHeartRate = preprocess(RestingHeartRate)
    RestingHeartRate['RestingHeartRate'] = RestingHeartRate.value
    RestingHeartRate = RestingHeartRate.sort_values('creationDate')

    VO2_df = pd.read_csv(f'{file_loction}/VO2Max.csv')
    VO2_df = preprocess(VO2_df)

    hrvdf = pd.read_csv(f"{file_loction}/HeartRateVariabilitySDNN.csv")
    hrvdf = preprocess(hrvdf)

    return StepCount,steps_by_date,RestingHeartRate,VO2_df, hrvdf



avg_rhrR=[76,69,68,60,54]

def get_score(range_val, val):
    for i in range(len(range_val)):
        if val >= range_val[i]:
            return i+1
    return 5

avg_daily_stepsR=[1000,4000,7000,9999]
avg_vo2maxR=[31.5,42,42.5,47]
avg_hrvR=[25,40,60,80]

def get_score_reverse(range_val, val):
    for i in range(len(range_val)):
        if val <= range_val[i]:
            return i+1
    return 5


def plot_boxplot_remove_minor(df, grp, val, ratio):
    sample = df[grp].value_counts(normalize=True)  # .plot.bar();
    sample = sample[sample > ratio].index

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    val_cnt = df[grp].value_counts(normalize=True)
    sns.barplot(x=val_cnt.index, y=val_cnt, ax=axes[0]).set_title(f'{val} sample each {grp}');
    if ratio > 0:
        sns.boxplot(x=grp, y=val, data=df[df[grp].isin(sample)], palette="Set3", ax=axes[1]).set_title(
            f'{val} distribution by {grp}');
    else:
        sns.boxplot(x=grp, y=val, data=df, palette="Set3", ax=axes[1]).set_title(f'{val} distribution by {grp}');
    return fig


def app():
    st.markdown('''
        # **Cardiovascular**
        ---
        ''')

    #st.sidebar.title("**Cardiovascular Index**")
    StepCount,steps_by_date,RestingHeartRate,VO2_df,HRV_df=load_data()

    avg_daily_steps=np.round(steps_by_date.Steps.mean())
    avg_vo2max=np.round(VO2_df.VO2Max.mean())
    avg_rhr=np.round(RestingHeartRate.RestingHeartRate.mean())
    avg_hrv=np.round(HRV_df['HeartRateVariabilitySDNN'].mean())

    pm1, pm2, pm3, pm4 = st.columns((1, 1, 1,1))

    pm1.markdown('**Daily Steps**')
    pm1.subheader(avg_daily_steps)

    pm2.markdown('**Vo2Max**')
    pm2.subheader(avg_vo2max)
    pm3.markdown('**Resting Heart Rate**')
    pm3.subheader(f'{avg_rhr}')

    pm4.markdown('**Heart Rate Variability**')
    pm4.subheader(f'{avg_hrv}')

    time_bins = [0, 4, 6, 9, 12, 18, 24]

    time_labels = ['00:00-03:59', '04:00-05:59', '06:00-08:59', '09:00-11:59', '12:00-17:59', '18:00-23:59']



    avg_daily_steps_Score=get_score_reverse(avg_daily_stepsR,avg_daily_steps)
    avg_vo2max_Score=get_score_reverse(avg_vo2maxR,avg_vo2max)
    avg_hrv_Score = get_score_reverse(avg_hrvR, avg_hrv)

    avg_rhr_Score=get_score(avg_rhrR,avg_rhr)

    import pandas as pd
    df = pd.DataFrame(dict(
        score=[avg_daily_steps_Score, avg_vo2max_Score, avg_rhr_Score,avg_hrv_Score],
        metric=['Daily Steps', 'Vo2Max', 'Resting Heart Rate','HeartRateVariability'
                ]))

    Redarfig = px.line_polar(df, r='score', theta='metric', line_close=True)
    st.markdown("""---""")

    st.subheader("Metrices Score :")
    st.plotly_chart(Redarfig, use_container_width=True)

    st.markdown("""---""")
    HRV_df['Time_Bin'] = pd.cut(HRV_df.hour, time_bins, labels=time_labels, right=False)
    HRV_df['Time_Bin']=HRV_df['Time_Bin'].astype(str)

    hrv_box_fig = px.box(HRV_df, x="Time_Bin", y="HeartRateVariabilitySDNN")
    hrv_box_fig.update_xaxes(categoryorder='array', categoryarray=time_labels)
    hrv_box_fig.update_traces(marker_color='#264653')
    hrv_box_fig.update_layout(title_text="Heart Rate Variability Distribution by hours", title_x=0,
                          margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Time Range/Hours",
                          yaxis_title="HeartRateVariability")
    hrv_box_fig.update_layout(boxmode="overlay")
    st.plotly_chart(hrv_box_fig)

    # hrv_box_week_fig = px.box(HRV_df, x="Time_Bin", y="HeartRateVariabilitySDNN",color="weekday")
    # hrv_box_week_fig.update_xaxes(categoryorder='array', categoryarray=time_labels)
    # hrv_box_week_fig.update_traces(marker_color='#264653')
    # hrv_box_week_fig.update_layout(title_text="Heart Rate Variability Distribution by hours/weekday", title_x=0,
    #                           margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Time Range/Hours",
    #                           yaxis_title="HeartRateVariability" )
    #                           yaxis_title="HeartRateVariability" )
    # st.plotly_chart(hrv_box_week_fig)


    daily_hrv = HRV_df.groupby('date')['HeartRateVariabilitySDNN'].mean().reset_index(name='HeartRateVariability')

    daily_hrv['HRV_Week'] = daily_hrv.HeartRateVariability.rolling(window=7, center=True).mean()
    daily_hrv['HRV_Month'] = daily_hrv.HeartRateVariability.rolling(window=30, center=True).mean()
    hrv_fig = daily_hrv.plot(x='date', y=['HRV_Week', 'HRV_Month'], title='Heart Rate Variablilty Weekly/Monthly Trends');
    st.plotly_chart(hrv_fig)
    # rhr_fig.update_traces(marker_color='#264653')
    # rhr_fig.update_layout(title_text="Resting Heart Rate Over Time", title_x=0,
    #                       margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Date",
    #                       yaxis_title="RestingHeartRate", )


    st.markdown("""---""")

    RestingHeartRate['Time Bin'] = pd.cut(RestingHeartRate.hour, time_bins, labels=time_labels, right=False)
    RestingHeartRate['Time Bin']=RestingHeartRate['Time Bin'].astype(str)

    rhr_box_fig = px.box(RestingHeartRate, x="Time Bin", y="RestingHeartRate")
    rhr_box_fig.update_xaxes(categoryorder='array', categoryarray=time_labels)
    rhr_box_fig.update_traces(marker_color='#264653')
    rhr_box_fig.update_layout(title_text="RestingHeartRate Distribution by hours", title_x=0,
                              margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Time Range/Hours",
                              yaxis_title="RestingHeartRate", )
    st.plotly_chart(rhr_box_fig)

    # rhr_box_week_fig = px.box(RestingHeartRate, x="Time Bin", y="RestingHeartRate",color="weekday")
    # rhr_box_week_fig.update_xaxes(categoryorder='array', categoryarray=time_labels)
    # rhr_box_week_fig.update_traces(marker_color='#264653')
    # rhr_box_week_fig.update_layout(title_text="RestingHeartRate Distribution by hours/Week", title_x=0,
    #                           margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Time Range/Hours",
    #                           yaxis_title="RestingHeartRate" )
    # st.plotly_chart(rhr_box_week_fig)



    rhr_fig = px.line(RestingHeartRate, x="startDate", y=['RestingHeartRate'])
    rhr_fig.update_traces(marker_color='#264653')
    rhr_fig.update_layout(title_text="Resting Heart Rate Over Time", title_x=0,
                          margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Date",
                          yaxis_title="RestingHeartRate", )
    st.plotly_chart(rhr_fig)
    st.markdown("""---""")
    st.write("Resting Heart Rate Distribution by hour")
    st.pyplot(plot_boxplot_remove_minor(RestingHeartRate, 'hour', 'RestingHeartRate', 0.01));

    st.write("Resting Heart Rate Distribution by Weekday")
    st.pyplot(plot_boxplot_remove_minor(RestingHeartRate, 'weekday', 'RestingHeartRate', 0));


    st.markdown("""---""")

    steps_by_date['weekly'] = steps_by_date.Steps.rolling(window=7, center=True).mean()
    steps_by_date['monthly'] = steps_by_date.Steps.rolling(window=30, center=True).mean()
    roll_step = steps_by_date.plot(x='date', y=['weekly', 'monthly']);
    roll_step.update_layout(title_text="Daily step counts rolling mean over week and month", title_x=0, margin=dict(l=0, r=10, b=10, t=30),
                            xaxis_title="Date",
                            yaxis_title="Steps",
                            )


    st.plotly_chart(roll_step)

    hour_steps_df = StepCount.groupby(['hour', 'date'])['value'].sum().reset_index(name='Steps')
    stephour = hour_steps_df.groupby(['hour'])['Steps'].mean().plot.bar();
    stephour.update_layout(title_text="Average Steps/Active hours", title_x=0,
                            margin=dict(l=0, r=10, b=10, t=30),xaxis_title="Hour",
                            yaxis_title="Steps",)
    st.plotly_chart(stephour)

    week_steps_df = StepCount.groupby(['weekday', 'date'])['value'].sum().reset_index(name='Steps')
    stepweek = week_steps_df.groupby(['weekday'])['Steps'].mean().plot.bar();
    stepweek.update_layout(title_text="Average Steps/Active Week", title_x=0,
                           margin=dict(l=0, r=10, b=10, t=30), xaxis_title="Hour",
                           yaxis_title="Steps", barmode='stack', xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(stepweek)



    st.write("Steps Distribution by Hours")
    st.pyplot(plot_boxplot_remove_minor(hour_steps_df, 'hour', 'Steps', 0.02));

    st.write("Steps Distribution by Weekday")
    st.pyplot(plot_boxplot_remove_minor(week_steps_df, 'weekday', 'Steps', 0));

    st.markdown("""---""")

    VO2_fig=VO2_df.plot.line( x="startDate", y=['VO2Max'])
    VO2_fig.update_traces(marker_color='#264653')
    VO2_fig.update_layout(title_text="Vo2Max Over Time", title_x=0,
                            margin=dict(l=0, r=10, b=10, t=30),xaxis_title="Date",
                            yaxis_title="Vo2Max")
    st.plotly_chart(VO2_fig)


    #steps_month_fig=plot_line_chart(StepCount.groupby(['month'])['value'].sum().reset_index(name='Steps'),
    #                chart_title='Number of Steps per month', y_label='Step Count', g_col='month')

    #st.plotly_chart(steps_month_fig)




