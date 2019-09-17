import sys
from pyspark import SparkContext
import time
import math
from collections import defaultdict

sc = SparkContext(appName="task_1")


def generate2pair(items):
    two_pairs = list()
    for i in items:
        for j in items:
            if j > i:
                if (i, j) not in two_pairs:
                    two_pairs.append((i, j))
    return two_pairs


def generate_pairs(freq_items, size):
    pairs = []
    freq = list(freq_items)
    for i in range(len(freq)):
        for j in range(i + 1, len(freq)):
            if (freq[j])[0:(size - 2)] == (freq[i])[0:(size - 2)]:
                item = list(set(freq[i]).union(set(freq[j])))
                pairs.append(item)
            else:
                break
    return pairs


def find_frequent_pairs(buckets, candidates, support):
    item_pairs = defaultdict(int)
    freq_pairs = defaultdict(str)
    for c in candidates:
        for key, value in buckets:
            if set(c).issubset(set(value)):
                if support == item_pairs[tuple(sorted(set(c)))]:
                    freq_pairs[tuple(sorted(set(c)))] = support
                    break
                else:
                    item_pairs[tuple(sorted(set(c)))] = item_pairs[tuple(sorted(set(c)))] + 1
    freq_pairs = sorted(freq_pairs)
    return freq_pairs


def apriori(buckets, support, support_part, no_of_buckets):
    items = {}
    bucket_count = 0
    buckets = list(buckets)
    for key, values in buckets:
        for v in values:
            if v in items:
                items[v] = items[v] + 1
            else:
                items[v] = 1
        bucket_count += 1
    freq_items1 = []
    for key, value in items.items():
        if value >= int(support_part):
            freq_items1.append(key)
    frequent_pairs = sorted(freq_items1)
    support_partition = math.floor(float(support)*(float(bucket_count)/float(no_of_buckets)))
    count = 2
    while len(freq_items1) != 0:
        if count == 2:
            new_candidate = generate2pair(freq_items1)
        elif count > 2:
            new_candidate = generate_pairs(freq_items1, count)
        count = count + 1
        freq_pairs = find_frequent_pairs(buckets, new_candidate, support_partition)
        frequent_pairs.extend(freq_pairs)
        freq_items1 = []
        for pair in freq_pairs:
            if pair not in freq_items1:
                freq_items1.append(pair)
        freq_items1 = sorted(freq_items1)
    return frequent_pairs


def count_occurrence(buckets, pairs):
    key_val = defaultdict(int)
    buckets = list(buckets)
    pairs = list(pairs)
    for item in pairs:
        for key, value in buckets:
            if type(item) == str:
                item = [item]
                i = tuple(list(item))
            else:
                i = tuple(item)
            if set(item).issubset(set(value)):
                if i not in key_val:
                    key_val[i] = 1
                else:
                    key_val[i] = key_val[i] + 1
    return key_val.items()


def write_to_file(candidate, frequent, file):
    candidate_pairs = candidate.map(lambda q: q[0]).collect()
    candidate_pairs = sorted(candidate_pairs, key=lambda q: (len(q), q))
    frequent_pairs = frequent.map(lambda q: q[0]).collect()
    frequent_pairs = sorted(frequent_pairs, key = lambda q: (len(q), q))
    result = open(file, 'w')
    result.write("Candidates: \n")
    count = len(candidate_pairs[0])
    for candidate in candidate_pairs:
        if len(candidate) == count and candidate != candidate_pairs[0]:
            result.write(", ")
        elif len(candidate) != 1:
            count = len(candidate)
            result.write("\n\n")
        if len(candidate) == 1:
            result.write("('"+str(candidate[0])+"')")
        else:
            result.write(str(candidate))
    result.write("\n\nFrequent Itemsets: \n")
    count = len(frequent_pairs[0])
    for pair in frequent_pairs:
        if len(pair) == count and pair != frequent_pairs[0]:
            result.write(", ")
        elif len(pair) != 1:
            count = len(pair)
            result.write("\n\n")
        if len(pair) == 1:
            result.write("('"+str(pair[0])+"')")
        else:
            result.write(str(pair))


def task1_son(case_no, support, input_file, output_file):
    data = sc.textFile(input_file)
    titles = data.first()
    filter_data = data.filter(lambda lines: lines != titles).map(lambda line: (line.split(',')))
    if case_no == 1:
        buckets = filter_data.map(lambda lines: (lines[0], lines[1])).combineByKey(lambda line: [line], lambda q, r: q + [r], lambda i, j: i + j)
    elif case_no == 2:
        buckets = filter_data.map(lambda lines: (lines[1], lines[0])).combineByKey(lambda line: [line], lambda q, r: q + [r], lambda i, j: i + j)
    support_partition = math.floor(int(support)/int(buckets.getNumPartitions()))
    if support_partition < 1:
        support_partition = 1
    no_of_buckets = buckets.count()
    # SON Phase-1
    map_phase1 = buckets.mapPartitions(lambda buck: apriori(buck, support, support_partition, no_of_buckets)).map(lambda q: (q, 1))
    reduce_phase1 = map_phase1.reduceByKey(lambda q, r: q).keys()
    reduce_phase1_distinct = reduce_phase1.distinct().collect()
    # SON Phase-2
    map_phase2 = buckets.mapPartitions(lambda bucket: count_occurrence(bucket, reduce_phase1_distinct))
    reduce_phase2 = map_phase2.reduceByKey(lambda q, r: q + r)
    reduce_phase2_filtered = reduce_phase2.filter(lambda q: int(q[1]) >= int(support))
    write_to_file(reduce_phase2, reduce_phase2_filtered, output_file)


if __name__ == '__main__':
    start_time = time.time()
    task1_son(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
    end_time = time.time()
    duration = end_time - start_time
    print("Duration: ", duration)