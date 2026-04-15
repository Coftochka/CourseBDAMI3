from dotenv import load_dotenv; 
load_dotenv()
import os
from moexalgo import session, Market



MOEX_TOKEN = os.getenv("MOEX_TOKEN")

session.TOKEN = MOEX_TOKEN

eq = Market('EQ')
data = eq.obstats(date='2024-10-15')


print(type(data))
print(data.head())
print(data.shape)

data.to_csv('obstats_2024-10-15.csv', index=False)
