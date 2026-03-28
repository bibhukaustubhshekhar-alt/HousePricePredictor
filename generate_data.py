import pandas as pd
import numpy as np

n = 1500
np.random.seed(42)
rows = []
for _ in range(n):
    bedrooms = np.random.randint(2, 6)
    bathrooms = np.random.randint(1, 5)
    sqft_living = np.random.randint(700, 4200)
    waterfront = int(np.random.choice([0, 1], p=[0.95, 0.05]))
    floors = int(np.random.choice([1, 2, 3]))
    price = 50000 + sqft_living * 200 + bedrooms * 10000 + bathrooms * 8000 + waterfront * 150000
    price = int(price * np.random.uniform(0.8, 1.2))
    rows.append([bedrooms, bathrooms, sqft_living, waterfront, floors, price])

raw = pd.DataFrame(rows, columns=['bedrooms','bathrooms','sqft_living','waterfront','floors','price'])
raw.to_csv('data/raw.csv', index=False)
processed = raw.drop(columns=['price']).sample(n=1500, random_state=42).reset_index(drop=True)
processed.to_csv('data/processed.csv', index=False)
print('wrote raw and processed with', len(raw), 'rows')