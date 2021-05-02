import matplotlib.pyplot as plt

def get_mining_profits():
    data_matrix = []
    dates = []
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
                dates.append(line.split(",")[0])
                total = 0
                count = 0
    data_matrix.reverse()
    return data_matrix, dates

def get_crypto_data():
    data = []
    lines = open("Binance_ETHUSDT_1h.csv", "r").read().splitlines()

    for line in lines[2:]:
        data.append(float(line.split(",")[8]))

    data.reverse()
    return data


m, d =get_mining_profits()
c = get_crypto_data()
print("length: " + str(len(m)) + " " + str ( len(c)))
for x in range(len(m)):
    print(d[x])
    print(m[x])
    print(c[x])


# plotting the points
plt.plot(d, m)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Mining rate over time')
plt.show()
plt.plot(d, c[:61])
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('volume over time')
plt.show()