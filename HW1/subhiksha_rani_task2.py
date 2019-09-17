from pyspark import SparkContext
import sys
import json
import time
sc = SparkContext(appName="task_2")

def part1(input1, input2, output_file):
    review_data = sc.textFile(input1)
    business_data = sc.textFile(input2)
    full_reviews = review_data.map(lambda x: json.loads(x))
    reviews = full_reviews.repartition(4)
    full_business = business_data.map(lambda y: json.loads(y))
    business = full_business.repartition(4)
    business_stars = reviews.map(lambda q: (q['business_id'], q['stars']))
    states = business.map(lambda q: (q['business_id'], q['state']))
    state_stars = states.join(business_stars).map(lambda x: x[1])
    start_val = (0, 0)
    sum_count = state_stars.aggregateByKey(start_val, lambda a, b: (a[0]+b, a[1]+1), lambda a,b: (a[0]+b[0], a[1]+b[1]))
    avg_stars = sum_count.map(lambda q: (q[0], q[1][0] / q[1][1])).sortBy(lambda r: (-r[1], r[0]))
    avg_state_stars = avg_stars.collect()
    with open(output_file, 'w') as opfile:
        opfile.write("state,stars\n")
        for key, value in avg_state_stars:
            opfile.write(str(key)+','+str(value)+'\n')
    return avg_stars


def part2(avg_stars, output_file):
    start_time1 = time.time()
    states1 = avg_stars.map(lambda q: q[0]).collect()
    print("m1: ", states1[:5])
    end_time1 = time.time()
    m1 = end_time1 - start_time1
    start_time2 = time.time()
    states2 = sc.parallelize(avg_stars.take(5)).map(lambda q: q[0]).collect()
    print("m2: ", states2)
    end_time2 = time.time()
    m2 = end_time2 - start_time2
    json_op = {}
    json_op["m1"] = m1
    json_op["m2"] = m2
    json_op["explanation"] = "From the m1 and m2 results we can see that m1 and m2 takes almost same time to execute. But m1 executes a little faster (by few milli seconds) than m2 in this case. So we can say that both m1 - collecting all the values and then printing first 5 out of it, and m2 - first take the 5 values and print it have almost similar efficiency."
    with open(output_file, 'w') as opfile:
        json.dump(json_op, opfile, indent=2)


if __name__ == '__main__':
    avg_stars = part1("review.json", "business.json", "op1.txt")
    part2(avg_stars, "op2.json")