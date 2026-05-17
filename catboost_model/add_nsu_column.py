import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

NSU_LAT = 54.8436
NSU_LON = 83.0941

df = pd.read_csv('final_output.csv')

df['distance_to_NSU_km'] = df.apply(
    lambda row: haversine(row['latitude'], row['longitude'], NSU_LAT, NSU_LON)
    if pd.notna(row['latitude']) and pd.notna(row['longitude']) else 999,
    axis=1
)

df.to_csv('final_output.csv', index=False)

print("Column distance_to_NSU_km has been added")
print(f"Total columns: {len(df.columns)}")
print(df.columns.tolist())