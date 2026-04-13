import altair as alt
import pandas as pd

try:
    df = pd.DataFrame({"Date": ["A", "B"], "Count": [10, 20]})
    chart = alt.Chart(df).mark_bar(
        color="#8b5cf6",
        cornerRadiusEnd=5,
        size=20
    ).encode(
        x="Date:O",
        y="Count:Q"
    )
    print("Bar chart successful")
    
    source = pd.DataFrame({
        "Status": ["Present", "Absent"],
        "Count": [15, 5]
    })
    donut = alt.Chart(source).mark_arc(innerRadius=60).encode(
        theta=alt.Theta("Count:Q", stack=True),
        color=alt.Color("Status:N")
    )
    print("Donut chart successful")
except Exception as e:
    print("Error:", e)
