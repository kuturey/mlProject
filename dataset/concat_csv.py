import pandas as pd

# каждый csv открывается в data frame, но нужно сначала запарсить
df1 = pd.read_csv('cian_flat_sale_1_100_novosibirsk_studio_extra.csv', sep=';')
df2 = pd.read_csv('cian_flat_sale_1_100_novosibirsk_rooms1_extra.csv', sep=';')
df3 = pd.read_csv('cian_flat_sale_1_100_novosibirsk_rooms2_extra.csv', sep=';')
df4 = pd.read_csv('cian_flat_sale_1_100_novosibirsk_rooms3_extra.csv', sep=';')
df5 = pd.read_csv('cian_flat_sale_1_100_novosibirsk_rooms4_extra.csv', sep=';')
df6 = pd.read_csv('cian_flat_sale_1_100_novosibirsk_rooms5_extra.csv', sep=';')

# конкатинация всех data frame в csv, минус описание
# df_concat = pd.concat([df1, df2, df3, df4, df5, df6], axis=0, ignore_index=True)
df_concat = pd.read_csv('flats_raw_extra.csv')
df_concat = df_concat.drop(['accommodation_type','author', 'deal_type', 'heating_type','house_material_type', 'house_number',
                            'location', 'object_type', 'phone', 'residential_complex', 'url'], axis=1)
df_concat.to_csv('flats_extra_upd.csv', index=False)
