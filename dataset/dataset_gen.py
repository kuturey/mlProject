import cianparser

# парсер с выбором локации и характеристик квартиры
novosibirsk_parser = cianparser.CianParser(location="Новосибирск")
data = novosibirsk_parser.get_flats(deal_type="sale", rooms=5, with_saving_csv=True, with_extra_data=True)

# сохранятеся в csv с названием по типу cian_flate_sale_1_100_novisibirsk_{date}