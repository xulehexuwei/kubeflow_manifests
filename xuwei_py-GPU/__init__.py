if __name__ == '__main__':
    with open('test.txt', 'w', encoding='utf8') as fp:
        for i in range(10):
            fp.write(f"{i}\n")

    import random
    with open('test.txt', "r") as f:
        data = f.read().splitlines()
        # 如果这里都爆内存的话，
        # 看起来只能使用文件指针，在getitem里边逐行读取了
        # 得到的data是 list[str]
    # random.shuffle(data)

    print(data)