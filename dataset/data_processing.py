import numpy as np
import pandas as pd
import re

def clean_numeric(value):
    if pd.isna(value):
        return np.nan

    value = str(value)
    
    value = re.sub(r'[^\d,.,-]', '', value)
    
    value = value.replace(',', '.')
    
    if not value or value == '':
        return np.nan
    
    try:
        return float(value)
    except:
        return np.nan
    
def fill_square_meters(df):
    remaining = df['total_meters'].isna().sum() + df['kitchen_meters'].isna().sum() + df['living_meters'].isna().sum()
    if remaining == 0:
        return df
    for i in range(len(df)):
        kitchen = df.loc[i, 'kitchen_meters']
        living = df.loc[i, 'living_meters']
        total = df.loc[i, 'total_meters']
        
        # 1-ый: есть оба, но нет общей
        if pd.notna(kitchen) and pd.notna(living) and pd.isna(total):
            df.loc[i, 'total_meters'] = kitchen + living
        
        # 2-ой: если есть кухня, но нет жилой
        elif pd.notna(kitchen) and pd.notna(total) and pd.isna(living):
            df.loc[i, 'living_meters'] = total - kitchen
        
        # 3-ий: если есть жилая, но нет кухни
        elif pd.notna(living) and pd.notna(total) and pd.isna(kitchen):
            df.loc[i, 'kitchen_meters'] = total - living
        
        # 4-ый: если нет кухни и жилой
        elif pd.notna(total) and pd.isna(kitchen) and pd.isna(living):
            df.loc[i, 'kitchen_meters'] = total * 0.20
            df.loc[i, 'living_meters'] = total * 0.80
            
        # 5-ый: есть только кухня, нет общей и жилой
        elif pd.notna(kitchen) and pd.isna(total) and pd.isna(living):
            df.loc[i, 'total_meters'] = kitchen / 0.20
            df.loc[i, 'living_meters'] = df.loc[i, 'total_meters'] - kitchen
            
        # 6-ой: есть только жилая, нет общей и кухни
        elif pd.notna(living) and pd.isna(total) and pd.isna(kitchen):
            df.loc[i, 'total_meters'] = living / 0.80
            df.loc[i, 'kitchen_meters'] = df.loc[i, 'total_meters'] - living
            
    return df

def get_era_house(row):
    if pd.notna(row['year_of_construction']):
        return row['year_of_construction']
    floors = row.get('floors_count', np.nan)
    
    era_map = [
        (floors <= 2, 1950, "Малоэтажные послевоенные"),
        (floors == 3, 1955, "Трехэтажные"),
        (floors == 4, 1957, "Четырехэтажные"),
        (floors == 5, 1963, "Хрущёвки"),
        (floors == 8, 1970, "Восьмиэтажные"),
        (floors == 9, 1975, "Брежневки"),
        (floors == 10, 1980, "Десятиэтажки"),
        (floors == 11, 1985, "Одиннацадтиэтажники"),
        (floors == 12, 1990, "Двенадцатиэтажники"),
        (floors == 13, 1995, "Тринадцатиэтажники"),
        (floors == 14, 2000, "Четырнадцатиэтажные"),
        (floors == 16, 2005, "Шестнадцатиэтажные"),
        (floors == 17, 2010, "Семнадцатиэтажные"),
        (floors == 18, 2015, "Восемнадцатиэтажные"),
        (floors == 19, 2018, "Девятнадцатиэтажные"),
        (floors == 20, 2020, "Двадцатиэтажные"),
        (floors == 21, 2021, "21-этажные"),
        (floors == 22, 2022, "22-этажные"),
        (floors == 23, 2023, "23-этажные"),
        (floors == 24, 2024, "24-этажные"),
        (floors == 25, 2025, "25-этажные"),
        (floors > 25, 2025, "Небоскребы"),
    ]
    
    for condition, year, era in era_map:
        if condition:
            return year
    return np.nan
    
def fill_year_construction(df):
    
    if df['year_of_construction'].notna().all():
        return df
    if 'district' in df.columns:
        df['year_of_construction'] = df.groupby('district')['year_of_construction'].transform(lambda x: x.fillna(x.median()))
        
    remaining = df['year_of_construction'].isnull().sum()
    mask = df['year_of_construction'].isna()
    if remaining > 0:
        df.loc[mask, 'year_of_construction'] = df[mask].apply(get_era_house, axis=1)
    remaining = df['year_of_construction'].isnull().sum()
    if remaining > 0:
        total_median = df['year_of_construction'].median()
        df['year_of_construction'] = df['year_of_construction'].fillna(total_median)
        
    df['year_of_construction'] = df['year_of_construction'].round().astype(int)
    
    return df

def fill_district(df):
    remaining = df['district'].isna().sum()
    if remaining == 0:
        return df
    if 'underground' in df.columns:
        metro_district = {}
        complete_metro = df[df['underground'].notna() & df['district'].notna()]
        for metro in complete_metro['underground'].unique():
            districts = complete_metro[complete_metro['underground'] == metro]['district'].mode()
            if not districts.empty:
                metro_district[metro] = districts[0]
            
        missing_district = df[df['district'].isna()]
        for ind, row in missing_district.iterrows():
            metro = row['underground']
            if pd.notna(metro) and metro in metro_district:
                df.loc[ind, 'district'] = metro_district[metro]
    remaining = df['district'].isna().sum()
    if remaining > 0 and 'street' in df.columns:
        street_district = {}
        complete_street = df[df['street'].notna() & df['district'].notna()]
        for street in complete_street['street'].unique():
            districts = complete_street[complete_street['street'] == street]['district'].mode()
            if not districts.empty:
                street_district[street] = districts[0]
        
        missing_district = df[df['district'].isna()]
        for idx, row in missing_district.iterrows():
            street = row['street']
            if pd.notna(street) and street in street_district:
                df.loc[idx, 'district'] = street_district[street]
    remaining = df['district'].isna().sum()
    if remaining > 0:
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        type_district = {}
        for htype in df['house_type'].dropna().unique():
            known = df[(df['house_type'] == htype) & (df['district'].notna())]
            if len(known) > 0:
                typical = known['district'].mode()
                if not typical.empty:
                    type_district[htype] = typical[0]
        missing_district = df[df['district'].isna()]
        for idx, row in missing_district.iterrows():
            htype = row['house_type']
            if pd.notna(htype) and htype in type_district:
                df.loc[idx, 'district'] = type_district[htype]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type'], axis=1)
    remaining = df['district'].isna().sum()
    if remaining > 0:
        df['district'] = df['district'].fillna(df['district'].mode()[0])
    return df
                
def fill_underground(df):
    remaining = df['underground'].isna().sum()
    if remaining == 0:
        return df
    if 'district' in df.columns:
        district_underground = {}
        complete_district = df[df['underground'].notna() & df['district'].notna()]
        for district in complete_district['district'].unique():
            undergrounds = complete_district[complete_district['district'] == district]['underground'].mode()
            if not undergrounds.empty:
                district_underground[district] = undergrounds[0]
        missing_underground = df[df['underground'].isna()]
        for idx, row in missing_underground.iterrows():
            district = row['district']
            if pd.notna(district) and district in district_underground:
                df.loc[idx, 'underground'] = district_underground[district]
    remaining = df['underground'].isna().sum()
    if remaining > 0 and 'street' in df.columns:
        street_underground = {}
        complete_street = df[df['street'].notna() & df['underground'].notna()]
        for street in complete_street['street'].unique():
            undergrounds = complete_street[complete_street['street'] == street]['underground'].mode()
            if not undergrounds.empty:
                street_underground[street] = undergrounds[0]
        
        missing_underground = df[df['underground'].isna()]
        for idx, row in missing_underground.iterrows():
            street = row['street']
            if pd.notna(street) and street in street_underground:
                df.loc[idx, 'underground'] = street_underground[street]
    return df

def fill_street(df):
    remaining = df['street'].isna().sum()
    if remaining == 0:
        return df
    if ('floors_count' in df.columns and 'year_of_construction' in df.columns 
        and 'district' in df.columns and 'underground' in df.columns):
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        df['location_key'] = (df['district'].astype(str) + '_' + df['underground'].astype(str) 
                              + '_' + df['house_type'].astype(str))
        combo_street = {}
        complete_combo = df[df['street'].notna() & df['location_key'].notna()]
        for key in complete_combo['location_key'].unique():
            streets = complete_combo[complete_combo['location_key'] == key]['street'].mode()
            if not streets.empty:
                combo_street[key] = streets[0]
        missing_streets = df[df['street'].isna() & df['location_key'].notna()]
        for idx, row in missing_streets.iterrows():
            key = row['location_key']
            if pd.notna(key) and key in combo_street:
                df.loc[idx, 'street'] = combo_street[key]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type', 'location_key'], axis=1)
    remaining = df['street'].isna().sum()
    if remaining > 0 and 'district' in df.columns and 'underground' in df.columns:
        df['district_metro'] = df['district'].astype(str) + '_' + df['underground'].astype(str)
        combo_street = {}
        complete_combo = df[df['street'].notna() & df['district_metro'].notna()]
        for combo in complete_combo['district_metro'].unique():
            streets = complete_combo[complete_combo['district_metro'] == combo]['street'].mode()
            if not streets.empty:
                combo_street[combo] = streets[0]
        missing_streets = df[df['street'].isna() & df['district_metro'].notna()]
        for idx, row in missing_streets.iterrows():
            combo = row['district_metro']
            if pd.notna(combo) and combo in combo_street:
                df.loc[idx, 'street'] = combo_street[combo]
        df = df.drop('district_metro', axis=1)
    remaining = df['street'].isna().sum()
    if (remaining > 0 and 'floors_count' in df.columns 
        and 'year_of_construction' in df.columns and 'underground' in df.columns):
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        df['location_key'] = (df['underground'].astype(str) 
                              + '_' + df['house_type'].astype(str))
        combo_street = {}
        complete_combo = df[df['street'].notna() & df['location_key'].notna()]
        for key in complete_combo['location_key'].unique():
            streets = complete_combo[complete_combo['location_key'] == key]['street'].mode()
            if not streets.empty:
                combo_street[key] = streets[0]
        missing_streets = df[df['street'].isna() & df['location_key'].notna()]
        for idx, row in missing_streets.iterrows():
            key = row['location_key']
            if pd.notna(key) and key in combo_street:
                df.loc[idx, 'street'] = combo_street[key]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type', 'location_key'], axis=1)
    remaining = df['street'].isna().sum()
    if (remaining > 0 and 'floors_count' in df.columns 
        and 'year_of_construction' in df.columns and 'district' in df.columns):
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        df['location_key'] = (df['district'].astype(str) 
                            + '_' + df['house_type'].astype(str))
        combo_street = {}
        complete_combo = df[df['street'].notna() & df['location_key'].notna()]
        for key in complete_combo['location_key'].unique():
            streets = complete_combo[complete_combo['location_key'] == key]['street'].mode()
            if not streets.empty:
                combo_street[key] = streets[0]
        missing_streets = df[df['street'].isna() & df['location_key'].notna()]
        for idx, row in missing_streets.iterrows():
            key = row['location_key']
            if pd.notna(key) and key in combo_street:
                df.loc[idx, 'street'] = combo_street[key]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type', 'location_key'], axis=1)
    remaining = df['street'].isna().sum()
    if (remaining > 0 and 'district' in df.columns):
        district_street = {}
        complete_district = df[df['street'].notna() & df['district'].notna()]
        for district in complete_district['district'].unique():
            streets = complete_district[complete_district['district'] == district]['street'].mode()
            if not streets.empty:
                district_street[district] = streets[0]
        missing_streets = df[df['street'].isna()]
        for idx, row in missing_streets.iterrows():
            district = row['district']
            if pd.notna(district) and district in district_street:
                df.loc[idx, 'street'] = district_street[district]
    remaining = df['street'].isna().sum()
    if (remaining > 0 and 'underground' in df.columns):
        underground_street = {}
        complete_underground= df[df['street'].notna() & df['underground'].notna()]
        for underground in complete_underground['underground'].unique():
            streets = complete_underground[complete_underground['underground'] == underground]['street'].mode()
            if not streets.empty:
                underground_street[underground] = streets[0]
        missing_streets = df[df['street'].isna()]
        for idx, row in missing_streets.iterrows():
            underground = row['underground']
            if pd.notna(underground) and underground in underground_street:
                df.loc[idx, 'street'] = underground_street[underground]
    return df     
 
def fill_author_type(df):
    remaining = df['author_type'].isna().sum()
    if remaining == 0:
        return df
    if ('floors_count' in df.columns and 'year_of_construction' in df.columns 
        and 'district' in df.columns and 'underground' in df.columns and 'street' in df.columns):
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        df['location_key'] = (df['district'].astype(str) + '_' + df['underground'].astype(str) 
                              + '_' + df['street'] + '_' + df['house_type'].astype(str))
        combo_author_type = {}
        complete_combo = df[df['author_type'].notna() & df['location_key'].notna()]
        for key in complete_combo['location_key'].unique():
            author_types = complete_combo[complete_combo['location_key'] == key]['author_type'].mode()
            if not author_types.empty:
                combo_author_type[key] = author_types[0]
        missing_author_types = df[df['author_type'].isna() & df['location_key'].notna()]
        for idx, row in missing_author_types.iterrows():
            key = row['location_key']
            if pd.notna(key) and key in combo_author_type:
                df.loc[idx, 'author_type'] = combo_author_type[key]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type', 'location_key'], axis=1)
    remaining = df['author_type'].isna().sum()
    if (remaining > 0 and 'floors_count' in df.columns and 'year_of_construction' in df.columns 
    and 'district' in df.columns and 'street' in df.columns):
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        df['location_key'] = (df['district'].astype(str) + '_' 
                                + df['street'] + '_' + df['house_type'].astype(str))
        combo_author_type = {}
        complete_combo = df[df['author_type'].notna() & df['location_key'].notna()]
        for key in complete_combo['location_key'].unique():
            author_types = complete_combo[complete_combo['location_key'] == key]['author_type'].mode()
            if not author_types.empty:
                combo_author_type[key] = author_types[0]
        missing_author_types = df[df['author_type'].isna() & df['location_key'].notna()]
        for idx, row in missing_author_types.iterrows():
            key = row['location_key']
            if pd.notna(key) and key in combo_author_type:
                df.loc[idx, 'author_type'] = combo_author_type[key]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type', 'location_key'], axis=1)
    remaining = df['author_type'].isna().sum()
    if (remaining > 0 and 'floors_count' in df.columns and 'year_of_construction' in df.columns 
    and 'district' in df.columns):
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        df['location_key'] = (df['district'].astype(str) + '_' + df['house_type'].astype(str))
        combo_author_type = {}
        complete_combo = df[df['author_type'].notna() & df['location_key'].notna()]
        for key in complete_combo['location_key'].unique():
            author_types = complete_combo[complete_combo['location_key'] == key]['author_type'].mode()
            if not author_types.empty:
                combo_author_type[key] = author_types[0]
        missing_author_types = df[df['author_type'].isna() & df['location_key'].notna()]
        for idx, row in missing_author_types.iterrows():
            key = row['location_key']
            if pd.notna(key) and key in combo_author_type:
                df.loc[idx, 'author_type'] = combo_author_type[key]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type', 'location_key'], axis=1)
    remaining = df['author_type'].isna().sum()
    if remaining > 0 and 'floors_count' in df.columns and 'year_of_construction' in df.columns:
        if 'house_type' not in df.columns:
            df['era'] = df.apply(get_era_house, axis=1)
            df['era_group'] = pd.cut(df['era'], 
                                        bins=[1940, 1960, 1980, 2000, 2010, 2025], 
                                        labels=['oldest', 'old', 'mid', 'new', 'newest'])
            df['floor_group'] = pd.cut(df['floors_count'],
                                        bins=[2, 5, 9, 12, 17, 50],
                                        labels=['малоэтажные', 'хрущевки', 'брежневки', 'высокие', 'очень высокие'])
            df['house_type'] = df['era_group'].astype(str) + '_' + df['floor_group'].astype(str)
        combo_author_type = {}
        complete_combo = df[df['author_type'].notna() & df['house_type'].notna()]
        for key in complete_combo['house_type'].unique():
            author_types = complete_combo[complete_combo['house_type'] == key]['author_type'].mode()
            if not author_types.empty:
                combo_author_type[key] = author_types[0]
        missing_author_types = df[df['author_type'].isna() & df['house_type'].notna()]
        for idx, row in missing_author_types.iterrows():
            key = row['house_type']
            if pd.notna(key) and key in combo_author_type:
                df.loc[idx, 'author_type'] = combo_author_type[key]
        df = df.drop(['era', 'era_group', 'floor_group', 'house_type'], axis=1)
        remaining = df['author_type'].isna().sum()
        if remaining > 0:
            mode_type = df['author_type'].mode()
            if not mode_type.empty:
                df['author_type'] = df['author_type'].fillna(mode_type[0])
    return df
# чтение данных
df = pd.read_csv('flats_extra_upd.csv')

# разбиение данных на числовые и категориальные
numeric_cols = ['floor','floors_count','kitchen_meters','living_meters','price','rooms_count', 
                'total_meters', 'year_of_construction']

catecorial_cols = ['author_type', 'district', 'street',
                    'underground']

drop_cols = ['finish_type']
df = df.drop(drop_cols, axis=1)

total_one_minus = (df == -1).values.sum()

print('='*50)
print('ДО ОБРАБОТКИ')
print('='*50)
nan_per_column = df.isna().sum()
print(nan_per_column)
print(f"КОЛИЧЕСТВО -1 = {total_one_minus}")
print('='*50)

# обработка числовых признаков

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

df['rooms_count'] = df['rooms_count'].replace([np.nan, -1.0], 1.0)
df = df.replace(-1.0, np.nan)

# nan_per_column = df.isnull().sum()
# print(nan_per_column)

df = fill_square_meters(df)
df = fill_year_construction(df)

# обработка категорильных признаков

df = fill_district(df)
df = fill_underground(df)
df = fill_street(df)
df = fill_author_type(df)

total_one_minus = (df == -1).values.sum()

print('ПОСЛЕ ОБРАБОТКИ')
print('='*50)
nan_per_column = df.isnull().sum()
print(nan_per_column)
print(f"КОЛИЧЕСТВО -1 = {total_one_minus}")
print('='*50)

df.to_csv('output.csv', index=False)



