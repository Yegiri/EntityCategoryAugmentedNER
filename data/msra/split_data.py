import random
import argparse

random.seed(2022)

# parser = argparse.ArgumentParser()
# parser.add_argument('--rate_of_data', required=True)

# args = parser.parse_args()
# rate = float(args.rate_of_data)/100

def split_data(mode, num):
    with open('{}/sentences.txt'.format(mode), 'r') as f:
        datas = f.readlines()
    with open('{}/tags.txt'.format(mode), 'r') as f:
        tags = f.readlines()
    
    selected_data = list(range(len(datas)))
    random.shuffle(selected_data)
    selected_data = selected_data[:num]
    data = [datas[i] for i in selected_data]
    tag =  [tags[i] for i in selected_data]
    print(len(data), '/', len(datas))

    with open('{}/sentences.txt'.format(mode), 'w') as f:
        for i in data:
            f.write(i)
    with open('{}/tags.txt'.format(mode), 'w') as f:
        for i in tag:
            f.write(i)

split_data('train', 5785)
split_data('val', 715)
split_data('test', 724)
