def get_mining_profits():
    data_matrix = []
    lines = open("mining_stats-2021-03-30.csv", "r").read().splitlines()
    length = len(lines)
    count = 0
    total = 0
    # the line fo the day to start from 277
    for line in lines[277:]:
        if line.split(",")[1] == '52':
            count += 1
            total += float(line.split(",")[4])
            if count % 12 == 0:
                # average profitablity for the hour
                data_matrix.append(total / 12)
                total = 0
                count = 0
    data_matrix.reverse()
    return data_matrix

def get_crypto_data():
    data = []
    lines = open("Binance_ETHUSDT_1h.csv", "r").read().splitlines()

    for line in lines[2:]:
        data.append(float(line.split(",")[8]))

    data.reverse()
    return data



# print(get_mining_profits())
# print(get_crypto_data())
