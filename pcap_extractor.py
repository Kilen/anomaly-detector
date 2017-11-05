import scapy.all as scapy
import numpy as np
import pandas as pd

# return np.array
def extract_data(path):
    print('------ extracting data:')
    raw_datas = []
    with scapy.PcapReader(path) as pcap_reader:
        i = 0
        for p in pcap_reader:
            ip = None
            if p.haslayer(scapy.IP):
                ip = p.getlayer(scapy.IP)
            item = {
                "no": i + 1,
                "time": p.time,
                "src": (ip and ip.src) or p.src,
                "sport": 0,
                "dst": (ip and ip.dst) or p.dst,
                "dport": 0
                # "len": len(scapy.corrupt_bytes(p)),
                # "protocal": p.summary().split('/')[2].split(' ')[1]
            }

            if ip and (p.haslayer(scapy.TCP) or p.haslayer(scapy.UDP)):
                item["sport"] = ip.sport
                item["dport"] = ip.dport
            raw_datas.append(item)

            i = i + 1
            if (i % 10000 == 1):
                print('extracting count: ', i)

    return np.array(raw_datas)

def entropy(data, key):
    x = np.array([item[key] for item in data])
    x_value_list = set(x)
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        ent -= p * np.log2(p)

    return ent

def split_by_interval(data, interval = 0.1):
    print('------ spliting and grouping:')
    grouped_data = []
    start = data[0]["time"]
    end = start + interval
    group = []
    i = 0
    for item in data:
        i = i + 1
        if item["time"] > end:
            grouped_data.append(np.array(group))
            group = []
            start = item["time"]
            end = start + interval
        group.append(item)
        if (i % 10000 == 1):
            print('spliting count: ', i)
    grouped_data.append(np.array(group))
    return np.array(grouped_data)
            

def extract_features(data):
    return {
        "count": data.shape[0],
        "src_ent": entropy(data, 'src'),
        "sport_ent": entropy(data, 'sport'),
        "dst_ent": entropy(data, 'dst'),
        "dport_ent": entropy(data, 'dport'),
        "beg_i": data[0]['no'],
        "end_i": data[-1]['no']
    }

def create_data_frame(grouped_data):
    print('------ processing groups:')
    df = pd.DataFrame(columns=[
        "beg_i", "end_i", "count", "src_ent",
        "sport_ent", "dst_ent", "dport_ent"
    ])
    i = 0
    for data in grouped_data:
        i = i + 1
        row = extract_features(data)
        df = df.append(row, ignore_index = True)
        if (i % 10 == 1):
            print('grouping count: ', i)

    print(df.describe())
    return df


raw_datas = extract_data('./data/sample_1000k.pcap')
grouped_data = split_by_interval(raw_datas)
df = create_data_frame(grouped_data)
df.to_csv("processed.csv", index = False)



