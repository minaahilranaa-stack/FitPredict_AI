
import pandas as pd

df = pd.read_csv("data/fitness_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nDuplicate Rows:", df.duplicated().sum())
assert not df.duplicated().any(), "Duplicate rows found!"
print("Duplicate User IDs:", df["User_ID"].duplicated().sum())

print("\nMissing Values by Column:")
print(df.isna().sum()[df.isna().sum() > 0])

assert df.shape == (15000, 25)
assert df["User_ID"].is_unique
assert df["Fitness_Level"].notna().all()
assert df["Age"].between(18, 75).all()
assert df["Exercise_Days_Per_Week"].between(0, 7).all()

print("\nTarget Distribution:")
print(df["Fitness_Level"].value_counts())

print("\nDataset validation passed!")
