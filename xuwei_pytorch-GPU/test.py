from tqdm import tqdm
import time

if __name__ == '__main__':
    for i in tqdm(range(100)):
        print(i)
        time.sleep(1)
