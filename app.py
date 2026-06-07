import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.joblib')
THRESHOLD = 0.3  # снижен ради recall: ловим больше уходящих (см. README)

st.title('Прогноз оттока клиента')
st.write('Введите данные клиента — модель оценит вероятность ухода.')

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider('Месяцев с компанией', 0, 72, 12)
    monthly = st.slider('Месячный платёж ($)', 18, 120, 65)
    contract = st.selectbox('Контракт', ['Month-to-month', 'One year', 'Two year'])
    internet = st.selectbox('Интернет', ['DSL', 'Fiber optic', 'No'])
with col2:
    techsup = st.selectbox('Тех. поддержка', ['No', 'Yes', 'No internet service'])
    security = st.selectbox('Онлайн-защита', ['No', 'Yes', 'No internet service'])
    payment = st.selectbox('Способ оплаты', ['Electronic check', 'Mailed check',
                           'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless = st.selectbox('Безбумажный счёт', ['Yes', 'No'])

if st.button('Предсказать'):
    X_new = pd.DataFrame([{
        'tenure': tenure, 'MonthlyCharges': monthly,
        'Contract': contract, 'InternetService': internet,
        'TechSupport': techsup, 'OnlineSecurity': security,
        'PaymentMethod': payment, 'PaperlessBilling': paperless
    }])
    proba = model.predict_proba(X_new)[0][1]
    st.metric('Вероятность ухода', f'{proba:.1%}')
    if proba >= THRESHOLD:
        st.error(f'Высокий риск (порог {THRESHOLD:.0%}) — клиента стоит удерживать.')
    else:
        st.success('Низкий риск оттока.')
    st.caption('Порог снижен до 30% намеренно: в задаче удержания '
               'пропустить уходящего дороже, чем ложная тревога.')
