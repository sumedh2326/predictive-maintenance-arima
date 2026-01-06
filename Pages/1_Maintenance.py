
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Maintenance", layout="wide")
st.title("🛠 Maintenance Schedule (Baseline Crossings)")

items = st.session_state.get("maintenance_items", [])
if not items:
    st.info("No items flagged yet.")
else:
    df = pd.DataFrame(items)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "maintenance_items.csv", "text/csv")
``
