import pandas as pd

# Excel файл унших
file_path = 'zzz.xlsx'
df = pd.read_excel(file_path)

# Бүх баганын нэрийг array (list) болгох
columns = df.columns.tolist()

print(columns)