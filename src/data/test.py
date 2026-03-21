from dotenv import load_dotenv; 
load_dotenv()
import os
from moexalgo import session, Market



MOEX_TOKEN = os.getenv("MOEX_TOKEN")

session.TOKEN = MOEX_TOKEN

eq = Market('EQ')
data = eq.obstats(date='2024-10-15')


# Теперь data - это DataFrame со всеми методами pandas
print(type(data))  # <class 'pandas.DataFrame'>
print(data.head())  # Первые 5 строк
print(data.shape)   # Размерность (строки, колонки)

data.to_csv('obstats_2024-10-15.csv', index=False)