import sys
from pyspark import SparkContext
import time

sc = SparkContext(appName = "task1")


def jaccard_similarity(B2U_map, q):
    bus1 = q[0][0]
    bus2 = q[0][1]
    user1 = set(B2U_map[bus1])
    user2 = set(B2U_map[bus2])
    jaccard_sim = float(len(user1.intersection(user2)) / len(user1.union(user2)))
    return (((bus1, bus2), jaccard_sim))


def minhashing(a, m, x):
    minNum = [min((ax * x + 1) % m for x in x[1]) for ax in a]
    return (x[0], minNum)


def interim(id1, rows, x):
    bands = int(len(x[1]) / rows)
    id1 = id1 + 1
    bus_id = x[0]
    signatures = x[1]
    bands_list = []
    idx = 0
    for b in range(0, bands):
        row = []
        for r in range(0, rows):
            row.append(signatures[idx])
            idx = idx + 1
        bands_list.append(((b, tuple(row)), [bus_id]))
        row.clear()
    return bands_list


def generate_candidate_pairs(q):
    business = q[1]
    business.sort()
    candidate_pairs = []
    for i in range(0, len(business)):
        for j in range(i + 1, len(business)):
            if j > i:
                candidate_pairs.append(((business[i], business[j]), 1))
    return candidate_pairs


def write_to_file(pairs, opfile):
    output = open(opfile, "w")
    output.write("business_id_1, business_id_2, similarity\n")
    final_values = []
    all_pairs = pairs.collect()
    for value in all_pairs:
        if value[0] < value[1][0]:
            final_values.append([value[0], value[1][0], value[1][1]])
        else:
            final_values.append([value[1][0], value[0], value[1][1]])
    sort_values = sorted(final_values)
    for triples in sort_values:
        output.write(str(triples[0]) + ", " + str(triples[1]) + ", " + str(triples[2]) + "\n")


def jaccard_based_lsh(input_file, output_file):
    data = sc.textFile(input_file)
    split_data = data.map(lambda line: line.split(','))
    filter_data = split_data.filter(lambda q: q[0] != "user_id").persist()
    users = filter_data.map(lambda q: q[0]).distinct().collect()
    businesses = filter_data.map(lambda q: q[1]).distinct().collect()
    B2U_map = filter_data.map(lambda q: (q[1], [q[0]])).reduceByKey(lambda a, b: a + b).collectAsMap()

    user_len = len(users)
    user_dict = {}
    for i in range(0, user_len):
        user_dict[users[i]] = i

    busi_len = len(businesses)
    bus_dict = {}
    for i in range(0, busi_len):
        bus_dict[businesses[i]] = i

    char_matrix = filter_data.map(lambda q: (q[1], [user_dict[q[0]]])).reduceByKey(lambda a, b: a + b)

    a = [1, 3, 9, 11, 13, 17, 19, 27, 29, 31, 33, 37, 39, 41, 43, 47, 51, 53, 57, 59, 61, 63, 65, 67, 69, 71, 73, 77, 79, 81, 83, 85, 87, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 269, 279, 289, 299]

    rows = 2
    signature_matrix = char_matrix.map(lambda q: minhashing(a, user_len, q))
    id1 = -1
    sig = signature_matrix.flatMap(lambda q: interim(id1, rows, q))
    generate_candidates = sig.reduceByKey(lambda a, b: a + b).filter(lambda q: len(q[1]) > 1)
    candidate_pairs = generate_candidates.flatMap(lambda q: generate_candidate_pairs(q)).distinct()
    jaccard_sim = candidate_pairs.map(lambda q: jaccard_similarity(B2U_map, q)).filter(lambda q: q[1] >= 0.5)
    jaccard_pairs = jaccard_sim.map(lambda q: (q[0][1], (q[0][0], q[1]))).sortByKey().map(lambda q: (q[1][0], (q[0], q[1][1]))).sortByKey()
    write_to_file(jaccard_pairs, output_file)


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    start_time = time.time()
    jaccard_based_lsh(input_file, output_file)
    end_time = time.time()
    print("Total time: ", end_time-start_time)