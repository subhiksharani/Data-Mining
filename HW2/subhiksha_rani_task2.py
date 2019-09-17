import sys
from pyspark import SparkContext
import time
from collections import defaultdict
from collections import Counter
from itertools import combinations

sc = SparkContext(appName="task2")


def generate2pair(items1):
    items1 = set(items1)
    two_pairs = []
    for item in combinations(items1, 2):
        item = list(item)
        item.sort()
        two_pairs.append(item)
    return two_pairs


def generate_pairs(freq_items, size):
    pairs = []
    len1 = len(freq_items) - 1
    len2 = len1 + 1
    for i in range(len1):
        a = freq_items[i]
        for j in range(i + 1, len2):
            b = freq_items[j]
            if a[0:(size - 2)] == b[0:(size - 2)]:
                pairs.append(list(set(freq_items[i]).union(set(freq_items[j]))))
            else:
                break
    return pairs


def find_frequent_pairs(buckets, candidates, support):
    item_count = {}
    for c in candidates:
        c = set(c)
        sort_c = sorted(c)
        tuple_c = tuple(sort_c)
        for value in buckets:
            if c.issubset(value):
                item_count[tuple_c] = item_count.get(tuple_c, 0) + 1
    freq_pairs = []
    for item in item_count:
        if item_count[item] >= support:
            freq_pairs.append(item)
    freq_pairs = sorted(freq_pairs)
    return freq_pairs


def apriori(all_buckets, support, no_of_buckets):
    frequent_pairs = list()
    buckets = list(all_buckets)
    support_threshold = float(support) * (float(len(buckets))/float(no_of_buckets))
    bucket_count = Counter()
    for b in buckets:
        bucket_count.update(b)
    freq_items1 = []
    for item in bucket_count:
        if bucket_count[item] >= int(support_threshold):
            freq_items1.append(item)
    freq_items1 = sorted(freq_items1)
    frequent_pairs.extend(freq_items1)
    cand_pairs_2 = generate2pair(freq_items1)
    freq_pair_2 = find_frequent_pairs(buckets, cand_pairs_2, support_threshold)
    frequent_pairs.extend(freq_pair_2)
    count = 3
    frequent_items = freq_pair_2
    while len(frequent_items) != 0:
        cand_pairs = generate_pairs(frequent_items, count)
        frequent_items = find_frequent_pairs(buckets, cand_pairs, support_threshold)
        frequent_pairs.extend(frequent_items)
        frequent_items.sort()
        count += 1
    return frequent_pairs


def count_occurrence(buckets, pairs):
    key_val = defaultdict(int)
    buckets = list(buckets)
    pairs = list(pairs)
    for item in pairs:
        if type(item) == str:
            item = [item]
            i = tuple(sorted(item))
        else:
            i = tuple(item)
        item = set(item)
        for value in buckets:
            if item.issubset(value):
                key_val[i] = key_val.get(i, 0) + 1
    return key_val.items()


def write_to_file(candidate, frequent, file):
    candidate_pairs = candidate.map(lambda q: q[0]).collect()
    candidate_pairs = sorted(candidate_pairs, key=lambda q: (len(q), q))
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
    frequent_pairs = frequent.map(lambda q: q[0]).collect()
    frequent_pairs = sorted(frequent_pairs, key=lambda q: (len(q), q))
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


def task2_son(filter_threshold, support, input_file, output_file):
    data = sc.textFile(input_file)
    titles = data.first()
    filter_data = data.filter(lambda lines: lines != titles).map(lambda line: (line.split(',')))
    buckets = filter_data.map(lambda lines: (lines[0], lines[1])).combineByKey(lambda line: [line], lambda q, r: q + [r], lambda i, j: i + j).filter(lambda q: len(q[1]) > filter_threshold).map(lambda record: set(record[1]))
    no_of_buckets = buckets.count()
    # SON Phase-1
    map_phase1 = buckets.mapPartitions(lambda buck: apriori(buck, support, no_of_buckets)).map(lambda q: (q, 1))
    reduce_phase1 = map_phase1.reduceByKey(lambda q, r: q).keys()
    reduce_phase1_distinct = reduce_phase1.distinct().collect()
    # SON Phase-2
    map_phase2 = buckets.mapPartitions(lambda bucket: count_occurrence(bucket, reduce_phase1_distinct))
    reduce_phase2 = map_phase2.reduceByKey(lambda q, r: (q + r))
    reduce_phase2_filtered = reduce_phase2.filter(lambda q: int(q[1]) >= int(support))
    write_to_file(reduce_phase2, reduce_phase2_filtered, output_file)


if __name__ == '__main__':
    start_time = time.time()
    task2_son(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
    end_time = time.time()
    duration = end_time - start_time
    print("Duration: ", duration)
    sc.stop()