import json
from json import encoder
import time
import numpy as np
import tensorflow_datasets as tfds

def Amazon_DataSet():
    types = ["books", "dvd", "electronics", "kitchen"]
    senti = ["negative", "positive"]
    data = {}
    for t in types:
        data[t] = {"combined":{"X":[], "Y":[]}}
        final = []
        for sentiment in senti:
            print(f"\nWorking with: {t}")
            print(f"{sentiment}")
            data[t][sentiment[0]] = {"temp":[], "X":[], "Y":[]}
            f = open(f"processed_acl/{t}/{sentiment}.review")
            temp_w = []
            t_count = 0
            total = 0
            for i, line in enumerate(f):
                line = line.replace("\n", "")
                line = line.split(" ")
                #negative = 0
                #label = 0
                #print(f"\n{i}) Negative")
                line = line[:-1]
                for j, item in enumerate(line):
                    item = item.split(":")
                    item[1] = int(item[1])
                    #print(item)
                    final.append(item[0])
                    temp_w.append(item[0])
                    line[j] = item

                t_count = i + 1
                total += len(line)
                data[t][sentiment[0]]["temp"].append(line)

            print(f"Average word count: {total/t_count}")
            print(f"Word Count: {len(set(temp_w))}")
            f.close()

        final = sorted(list(set(final)))
        #print(final[:100])
        #input()

        print(f"Final Word Count: {len(set(final))}")
        for sentiment in senti:
            print(sentiment)
            temp_data = data[t][sentiment[0]].pop("temp")
            for i, line in enumerate(temp_data):
                print(i, len(temp_data))
                vector = [0 for i in range(0, len(final))]
                for item in line:
                    #print(item)
                    #input()
                    vector[final.index(item[0])] = item[1]
                    #print(vector[final.index(item[0])])
                    #input()
                data[t][sentiment[0]]["X"].append(vector)
                data[t]["combined"]["X"].append(vector)

                val = None
                if sentiment == "positive":
                    val = 1
                else:
                    val = 0

                data[t][sentiment[0]]["Y"].append(val)
                data[t]["combined"]["Y"].append(val)

        f = open("amazon_data.json", "w")
        json.dump(data, f, indent=4)
        f.close()
    return

def IMDB_8K_Data():
    dataset, info = tfds.load("imdb_reviews/subwords8k", with_info=True)
    print(info.features["text"].encoder)
    print(info.features["text"])
    print(info.features)
    print(info)
    print(info.features["text"].encoder.subwords)
    size = info.features["text"].encoder.vocab_size
    print(f"Vocab Size: {size}")
    print("Loading Training Data")
    temp_data = list(dataset["train"].as_numpy_iterator())
    print("Loading Testing Data")
    temp_data += list(dataset["test"].as_numpy_iterator())

    data = np.zeros((50_000, size+1), dtype="int64")
    #data_x = np.zeros((50_000, 8168), dtype="int64")
    #data_y = np.zeros((50_000, 1), dtype="int64")
    #print(data_y[:5])

    #temp = []
    for i, item in enumerate(temp_data):
        print(f"{i+1} out of 50 000")
        #data_y[i] = item["label"]
        #print(item["label"])
        #print(data[i][0])
        data[i][0] = item["label"]
        #temp.append(max(item["text"]))
        #temp.append(min(item["text"]))
        #print(item["text"])
        for val in item["text"]:
            #data_x[i][val] += 1
            data[i][val] += 1
        #print(data[i])
        #input()

    #print("Saving to file")
    #np.save("IMDB.npy", data)
    #print("Done saving!")

    print("Saving to compressed file")
    np.savez_compressed("IMDB.npz", data)
    print("Done saving!")
    
    # MIN is 1
    #print(min(temp))
    # MAX is 8168
    #print(max(temp))


    #print(x[0])
    #y = np.array(list(dataset["train"].as_numpy_iterator()))[:, 1]
    #print(x[0])
    #print(dataset['train'].shape())
    #X, Y = tfds.as_numpy(tfds.load("imdb_reviews/subwords8k", split="train", shuffle_files=False))

    return

def IMBD_Load():
    print("Testing Compressed Load")
    t = time.perf_counter()
    data = np.load("IMDB.npy")
    print(f"Time to load: {time.perf_counter()-t:.4f}s")
    print("Getting Data")
    t = time.perf_counter()
    print(data[0])
    print(f"Time to get data: {time.perf_counter()-t:.4f}s")
    print("Load Complete")

    print("\nTesting Compressed Load")
    t = time.perf_counter()
    data = np.load("IMDB.npz")
    print(f"Time to load: {time.perf_counter()-t:.4f}s")
    print("Getting Data")
    t = time.perf_counter()
    data = data["arr_0"]
    print(f"Time to get data: {time.perf_counter()-t:.4f}s")
    t = time.perf_counter()
    print(data[0])
    print(f"Time to get line in data: {time.perf_counter()-t:.4f}s")
    print("Load Complete")



if __name__ == "__main__":
    #Amazon_DataSet()
    IMDB_8K_Data()
    #IMBD_Load()