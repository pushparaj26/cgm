import CGM_DASH
import streamlit as st
import movement_dash


PAGES = {
    "Metabolic Index": CGM_DASH,
    "Cardiovascular Index":movement_dash
}

st.sidebar.image("HE_logo_H.png", width=200)
st.sidebar.title('Navigation')

selection = st.sidebar.radio("Go to Index", list(PAGES.keys()))
page = PAGES[selection]
page.app()