from pyspark import SparkContext
import sys
import json
from operator import add

def reviews(input_file, output_file):
    sc = SparkContext(appName="task_1")
    file = sc.textFile(input_file)
    full_data = file.map(lambda x: json.loads(x))
    data = full_data.repartition(4)
    useful_reviews = data.filter(lambda q: 0 < q['useful']).count()
    star_rating = data.filter(lambda q: 5.0 == q['stars']).count()
    review_len = data.map(lambda q: (len(q['text']), 1)).reduceByKey(add).takeOrdered(1, key=lambda r: -r[0])
    no_of_users = data.map(lambda q: (q['user_id'], 1)).reduceByKey(add).count()
    user_reviews = data.map(lambda q: (q['user_id'], 1)).reduceByKey(add).takeOrdered(20, key=lambda r:(-r[1], r[0]))
    no_business = data.map(lambda q: (q['business_id'], 1)).reduceByKey(add).count()
    business_reviews = data.map(lambda q: (q['business_id'], 1)).reduceByKey(add).takeOrdered(20, key=lambda r:(-r[1], r[0]))
    json_op = {}
    json_op["n_review_useful"] = useful_reviews
    json_op["n_review_5_star"] = star_rating
    for items in review_len:
        json_op["n_characters"] = items[0]
    json_op["n_user"] = no_of_users
    json_op["top20_user"] = list(map(list, user_reviews))
    json_op["n_business"] = no_business
    json_op["top20_business"] = list(map(list, business_reviews))
    with open(output_file, 'w') as opfile:
        json.dump(json_op, opfile, indent=2)
    exit()

if __name__ == '__main__':
    reviews(sys.argv[1], sys.argv[2])