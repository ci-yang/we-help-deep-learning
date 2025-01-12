import os
import json
import math
from urllib.request import urlopen

def get_api_data(path):
    response = urlopen(path)
    data = json.loads(response.read().decode('utf-8'))
    return data

def clean_product_data(item):
    return {
        'id': item['Id'],
        'describe': item['Describe'],
        'is_i5': 'i5處理器' in item['Describe'],
        'price': item['Price'],
        'rating': item['ratingValue'] if item['ratingValue'] else -1,
        'review': item['reviewCount'] if item['reviewCount'] else 0
    }

def get_product_data(page):
    product_data_list = []
    page_attr = '&attr=&page='
    for i in range(1, page+1):
        path = f'{url}{page_attr}{i}'
        data = get_api_data(path)
        product_data_list = product_data_list + data['Prods']
    return product_data_list

def create_product_file(file_path, produect_id_list):
    try:
        with open(file_path, 'w',  encoding='utf-8') as file:
            for product_id in produect_id_list:
                file.write(f'{product_id}\n')
    except Exception as e:
        print(f"creating file error: {e}")

def execute_task1(product_list):
    print('Creating Task1 Result...')
    file_name = 'products.txt'
    product_id_list = [product.get('id') for product in product_list]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, file_name)
    create_product_file(output_path, product_id_list)
    print(f'Task1 Done, please see the {file_name} in {os.path.dirname(output_path)}')

def execute_task2(product_list):
    print('Creating Task2 Result...')
    file_name = 'best-products.txt'
    filtered_product_list = list(filter(
        lambda product: product.get('review', 0) >= 1 and product.get('rating', -1) > 4.9, 
        product_list
    ))
    product_id_list = [product.get('id') for product in filtered_product_list]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, file_name)
    create_product_file(output_path, product_id_list)
    print(f'Task2 Done, please see the {file_name} in {os.path.dirname(output_path)}')

def execute_task3(product_list):
    print('Creating Task3 Result...')
    filtered_product_list = list(filter(
        lambda product: product.get('is_i5') == True,
        product_list
    ))
    product_price_list = [product.get('price') for product in filtered_product_list]
    result = round(sum(product_price_list) / len(product_price_list), 2)
    print(f'Task3 Result: {result}')

def execute_task4(product_list):
    print('Creating Task4 Result...')
    file_name = 'standardization.csv'
    
    def calculate_mean(numbers):
        return sum(numbers) / len(numbers)

    def calculate_std(numbers, mean):
        squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
        return math.sqrt(squared_diff_sum / len(numbers))

    def create_csv(file_path, results):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('id,price,z-score\n')
                for result in results:
                    file.write(f"{result['id']},{result['price']},{result['z_score']}\n")
        except Exception as e:
            print(f"creating file error: {e}")
    
    prices = [product.get('price') for product in product_list]
    mean = calculate_mean(prices)
    std = calculate_std(prices, mean)
    
    results = []
    for product in product_list:
        price = product['price']
        z_score = (price - mean) / std
        results.append({
            'id': product['id'],
            'price': price,
            'z_score': round(z_score, 2)
        })
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, file_name)
    create_csv(output_path, results)
    print(f'Task4 Done, please see the {file_name} in {os.path.dirname(output_path)}')

if __name__ == '__main__':
    # 1. get data by calling API
    # 2. get total page and calling API each page and getting product data
    # 3. clean product data to fit all requirement of tasks
    # 4. output the responding result of the task
    cate_id = 'DSAA31'
    url = f'https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid={cate_id}'
    data = get_api_data(url)
    total_page = data['TotalPage']
    product_data_list = get_product_data(total_page)
    product_list = [clean_product_data(product) for product in product_data_list]

    execute_task1(product_list)
    execute_task2(product_list)
    execute_task3(product_list)
    execute_task4(product_list)