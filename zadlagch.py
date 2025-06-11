import pandas as pd

file_path = "ЗТО_2020-2024_ашиглах_final.xlsx"
df = pd.read_excel(file_path)

# Олон баганын нэрийг жагсаалтаар оруулна
cols = [
    "Авто зам - Замын харьяалал",
    "Авто зам - Замын ангилал",
    "Авто зам - Замын гадаргуу",
    "Авто зам - Замын онцлог",
    "Авто зам - Үзэгдэх орчин",
    "Авто зам - Цаг агаар",
    "Авто зам - Бусад",
    "Авто зам - Замын хэсэг",
    "Авто зам - Ослын ноцтой байдал",
    "Авто зам - Осолд нөлөөлөх хүчин зүйл",
    "Авто зам - Ослын нөхцөл",
]  # шаардлагатай бүх баганын нэр

for col in cols:
    values = df[col].dropna().unique()
    for val in values:
        new_col = f"{col} {val}"
        df[new_col] = (df[col] == val).astype(int)

# Үр дүнг Excel файлд хадгалах
df.to_excel("output_onehot.xlsx", index=False)
