import json

data = json.load(open('data/facturas/fg.json'))
print(f'Tipo: {type(data)}')
print(f'Longitud: {len(data)}')

for i, item in enumerate(data[:5]):
    print(f'\nElemento {i}:')
    print(f'  Tipo: {type(item)}')
    if isinstance(item, dict):
        print(f'  Keys: {list(item.keys())}')
        if 'type' in item:
            print(f'  type: {item["type"]}')
        if 'data' in item:
            print(f'  data tiene {len(item["data"])} elementos')
            if len(item['data']) > 0:
                print(f'  Primer elemento de data: {type(item["data"][0])}')
                if isinstance(item['data'][0], dict):
                    print(f'  Keys del primer elemento: {list(item["data"][0].keys())[:10]}...')
