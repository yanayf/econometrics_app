import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
import statsmodels.stats.outliers_influence as smoi
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# Set app title in Hebrew
st.set_page_config(page_title="אפליקציה להוראת אקונומטריקה")

st.title("📊 אפליקציה ללימוד אקונומטריקה בסיסית")

uploaded_file = st.file_uploader("📂 העלו קובץ CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 תצוגה מקדימה של הנתונים")
    st.dataframe(df.head())

    action = st.selectbox("📌 מה תרצו לעשות?", ["סטטיסטיקה תיאורית", "ניתוח גרפי", "להריץ רגרסיה"])

    if action == "סטטיסטיקה תיאורית":
        st.write("📊 **סטטיסטיקה תיאורית של הנתונים**")
        st.write(df.describe())

    elif action == "ניתוח גרפי":
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("📌 בחרו משתנה לתצוגה גרפית", numeric_columns)
            fig, ax = plt.subplots()
            ax.hist(df[selected_column], bins=20, edgecolor="black")
            st.pyplot(fig)

    elif action == "להריץ רגרסיה":
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_columns) >= 2:
            y_var = st.selectbox("🎯 בחרו משתנה תלוי (Y)", numeric_columns)
            x_vars = st.multiselect("📌 בחרו משתנים מסבירים (X)", [col for col in numeric_columns if col != y_var])

            if x_vars:
                X = df[x_vars]
                X = sm.add_constant(X)
                Y = df[y_var]

                model = sm.OLS(Y, X).fit()
                st.text(model.summary())

                coef_dict = model.params.to_dict()
                equation = f"{y_var} = {coef_dict['const']:.3f}"
                for var in x_vars:
                    equation += f" + {coef_dict[var]:.3f}*{var}"
                st.latex(equation)

                alpha = st.slider("⚖️ בחרו רמת מובהקות (α)", 0.01, 0.10, 0.05, 0.01)
                p_values = model.pvalues
                st.write("📌 **בדיקות מובהקות (ערכי p)**")
                st.write(p_values)

                st.write("📌 **בדיקת מובהקות כללית (F-test)**")
                if model.f_pvalue < alpha:
                    st.write(f"p-value של מבחן F: {model.f_pvalue:.5f} (המודל מובהק ברמת α={alpha})")

                st.write("📌 **בדיקת מולטיקולינאריות (VIF)**")
                vif_data = pd.DataFrame()
                vif_data["משתנה"] = X.columns
                vif_data["VIF"] = [smoi.variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                st.write(vif_data)

                st.write("📌 **בדיקת הטרוסקדסטיות (Breusch-Pagan)**")
                bp_test = smd.het_breuschpagan(model.resid, model.model.exog)
                st.write(f"מבחן BP p-value: {bp_test[1]:.4f}")

                st.write("📌 **בדיקת נורמליות של השאריות**")
                fig, ax = plt.subplots()
                ax.hist(model.resid, bins=20, edgecolor="black")
                st.pyplot(fig)

                shapiro_test = stats.shapiro(model.resid)
                st.write(f"מבחן Shapiro-Wilk p-value: {shapiro_test[1]:.4f}")

                st.write("📌 **בדיקת מתאם סדרתי (Durbin-Watson)**")
                dw_stat = sm.stats.stattools.durbin_watson(model.resid)
                st.write(f"סטטיסטי Durbin-Watson: {dw_stat:.4f}")

st.write("📌 **האפליקציה הזו נועדה לעזור להבין רגרסיה ליניארית מסוג אומדי ריבועים פחותים ובדיקות מובהקות בצורה אינטראקטיבית**")
