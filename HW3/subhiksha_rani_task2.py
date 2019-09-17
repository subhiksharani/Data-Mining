import sys, time, math
from pyspark import SparkContext
from operator import add
from pyspark.mllib.recommendation import ALS, Rating

sc = SparkContext(appName = "task2")


def model_based_CF(input_file, test_file, output_file):
    start_time = time.time()
    data = sc.textFile(input_file)
    header = data.first()
    full_data = data.filter(lambda line: line != header).map(lambda line: (line.split(',')))
    users = full_data.map(lambda x: x[0]).collect()
    businesses = full_data.map(lambda x: x[1]).collect()
    user_dict = {}
    bus_dict = {}
    user_idx_dict = {}
    bus_idx_dict = {}
    for i, user in enumerate(users):
        user_dict[user] = i
        user_idx_dict[i] = user
    for i, business in enumerate(businesses):
        bus_dict[business] = i
        bus_idx_dict[i] = business
    ratings_data = full_data.map(lambda x: Rating(int(user_dict[x[0]]), int(bus_dict[x[1]]), float(x[2])))
    rank = 2
    numIterations = 5
    model = ALS.train(ratings_data, rank, numIterations)
    test_data = sc.textFile(test_file)
    test_header = test_data.first()
    test_full_data = test_data.filter(lambda line: line != test_header).map(lambda line: (line.split(',')))
    test_users = test_full_data.map(lambda x: x[0]).collect()
    test_businesses = test_full_data.map(lambda x: x[1]).collect()
    for index, user in enumerate(test_users):
        if user not in user_dict:
            while index in user_dict.values():
                index += 1
            user_dict[user] = index
            user_idx_dict[index] = user
    for index, business in enumerate(test_businesses):
        if business not in bus_dict:
            while index in bus_dict.values():
                index += 1
            bus_dict[business] = index
            bus_idx_dict[index] = business
    testing_rdd = test_full_data.map(lambda x: Rating(int(user_dict[x[0]]), int(bus_dict[x[1]]), float(x[2])))
    testing_data = testing_rdd.map(lambda x: (x[0], x[1]))
    pred_data = model.predictAll(testing_data).map(lambda x: ((x[0], x[1]), x[2])).cache()
    predictions = pred_data.map(lambda x: (x[0][0], x[0][1], x[1])).collect()
    outFile = open(output_file, "w")
    outFile.write('user_id, business_id, prediction\n')
    for pred in predictions:
        outFile.write(str(user_idx_dict[pred[0]]) + "," + str(bus_idx_dict[pred[1]]) + "," + str(pred[2]) + "\n")
    outFile.close()
    end_time = time.time()
    print("Model-Based Time: ", end_time - start_time)
    sc.stop()


def user_based_CF(input_file, test_file, output_file):
    start_time = time.time()
    data = sc.textFile(input_file)
    header = data.first()
    full_data = data.filter(lambda line: line != header).map(lambda line: (line.split(',')))
    test_data = sc.textFile(test_file)
    test_header = test_data.first()
    full_test_data = test_data.filter(lambda line: line != test_header).map(lambda line: (line.split(',')))
    ratings_by_user = full_data.map(lambda q: ((q[0]), ((q[1]), float(q[2])))).groupByKey().sortByKey(True)
    user_rating = ratings_by_user.mapValues(dict).collectAsMap()
    ratings_by_business = full_data.map(lambda q: ((q[1]), ((q[0]), float(q[2])))).groupByKey().sortByKey(True)
    business_rating = ratings_by_business.mapValues(dict).collectAsMap()
    user_rating_rdd = sc.broadcast(user_rating)
    business_rating_rdd = sc.broadcast(business_rating)

    def find_weight(user, business, business_rating, user_rating):
        business_rating_rdd = business_rating.value
        user_rating_rdd = user_rating.value
        if user in user_rating_rdd:
            calc_weight = 0
            user_bus_list = list(user_rating_rdd.get(user))
            user_bus = user_rating_rdd.get(user)
            user_rating_total = sum(user_bus.values())
            rating_weight = []
            bus_idx_1 = []
            bus_idx_2 = []
            user_avg = user_rating_total / len(user_bus)
            if business_rating_rdd.get(business) is None:
                return (user, business, str(user_avg))
            else:
                bus_usr_list = list(business_rating_rdd.get(business))
                if (len(bus_usr_list) != 0):
                    for i in range(0, len(bus_usr_list)):
                        rating_sum1 = 0
                        rating_sum2 = 0
                        usr_idx = 0
                        del bus_idx_1[:]
                        del bus_idx_2[:]
                        curr_bus_val = user_rating_rdd[bus_usr_list[i]].get(business)
                        while usr_idx < len(user_bus_list):
                            if user_rating_rdd[bus_usr_list[i]].get(user_bus_list[usr_idx]):
                                rating_sum1 += user_rating_rdd[user].get(user_bus_list[usr_idx])
                                rating_sum2 += user_rating_rdd[bus_usr_list[i]].get(user_bus_list[usr_idx])
                                bus_idx_1.append(user_rating_rdd[user].get(user_bus_list[usr_idx]))
                                bus_idx_2.append(user_rating_rdd[bus_usr_list[i]].get(user_bus_list[usr_idx]))
                            usr_idx += 1
                        if len(bus_idx_1) != 0:
                            avg_user1 = rating_sum1 / len(bus_idx_1)
                            avg_user2 = rating_sum2 / len(bus_idx_2)
                            numerator = 0
                            denominator_a = 0
                            denominator_b = 0
                            for j in range(0, len(bus_idx_1)):
                                diff1 = bus_idx_1[j] - avg_user1
                                diff2 = bus_idx_2[j] - avg_user2
                                numerator += diff1*diff2
                                denominator_a += pow(diff1, 2)
                                denominator_b += pow(diff2, 2)
                            denominator = math.sqrt(denominator_a) * math.sqrt(denominator_b)
                            if denominator != 0:
                                calc_weight = numerator / denominator
                            predicted_diff_weight = (curr_bus_val - avg_user2) * calc_weight
                            rating_weight.append((predicted_diff_weight, calc_weight))
                    pred_numerator = 0
                    pred_denominator = 0
                    for k in range(0, len(rating_weight)):
                        pred_numerator += rating_weight[k][0]
                        pred_denominator += abs(rating_weight[k][1])
                    prediction = -1
                    if pred_numerator == 0 or pred_denominator == 0:
                        prediction = user_avg
                        return (user, business, str(prediction))
                    else:
                        prediction = user_avg + (pred_numerator / pred_denominator)
                        if prediction < 0:
                            prediction = 0.0
                        elif prediction > 5:
                            prediction = 5.0
                        return (user, business, str(user_avg))
                else:
                    return (user, business, "2.7")
        else:
            return (user, business, str("2.7"))

    interim_weight = full_test_data.map(lambda q: find_weight(q[0], q[1], business_rating_rdd, user_rating_rdd))
    final_weight = interim_weight.collect()
    result = open(output_file, 'w')
    result.write("user_id, business_id, prediction" + "\n")
    size = len(final_weight)
    for i in range(0, size):
        result.write(str(final_weight[i][0]) + "," + str(final_weight[i][1]) + "," + str(final_weight[i][2]) + "\n")
    result.close()
    end_time = time.time()
    print("User-Based CF time: ", end_time-start_time)
    sc.stop()


def item_based_CF(input_file, test_file, output_file):
    start_time = time.time()
    data = sc.textFile(input_file)
    header = data.first()
    full_data = data.filter(lambda line: line != header).map(lambda line: (line.split(',')))
    all_users = full_data.map(lambda x: x[0]).collect()
    all_businesses = full_data.map(lambda x: x[1]).collect()
    user_dict = {}
    business_dict = {}
    user_idx_dict = {}
    bus_idx_dict = {}
    for i, user in enumerate(all_users):
        user_dict[user] = i
        user_idx_dict[i] = user
    for i, business in enumerate(all_businesses):
        business_dict[business] = i
        bus_idx_dict[i] = business
    train_data = full_data.map(lambda x: (int(user_dict[x[0]]), int(business_dict[x[1]]), float(x[2])))
    test_data = sc.textFile(test_file)
    test_header = test_data.first()
    full_test_data = test_data.filter(lambda line: line != test_header).map(lambda line: (line.split(',')))
    test_users = full_test_data.map(lambda x: x[0]).collect()
    test_businesses = full_test_data.map(lambda x: x[1]).collect()
    for id, user in enumerate(test_users):
        if user not in user_dict:
            while id in user_dict.values():
                id += 1
            user_dict[user] = id
            user_idx_dict[id] = user

    for id, business in enumerate(test_businesses):
        if business not in business_dict:
            while id in business_dict.values():
                id += 1
            business_dict[business] = id
            bus_idx_dict[id] = business

    test_data_map = full_test_data.map(lambda x: (int(user_dict[x[0]]), int(business_dict[x[1]]))).distinct()
    ratings_data = train_data.map(lambda q: ((q[0], q[1]), q[2])).reduceByKey(add).collectAsMap()
    avg_ratings = train_data.map(lambda q: (q[0], [q[2]])).reduceByKey(add)
    avg_items = train_data.map(lambda q: (q[1], [q[2]])).reduceByKey(add).map(lambda q: (q[0], float(sum(q[1]) / len(q[1])))).collectAsMap()
    avg_dict = avg_ratings.map(lambda q: (q[0], float(sum(q[1]) / len(q[1])))).collectAsMap()
    u2b_map = train_data.map(lambda q: (q[0], [q[1]])).reduceByKey(add).collectAsMap()
    b2u_map = train_data.map(lambda q: (q[1], [q[0]])).reduceByKey(add).collectAsMap()

    def predict_result(active_user, active_business, weights):
        if active_business in avg_items:
            avg = avg_items[active_business]
        elif active_user in avg_dict:
            avg = avg_dict[active_user]
        else:
            avg = 2.5
        if weights == 0:
            return avg
        neighbour_size = len(weights)
        if neighbour_size == 0:
            return avg
        if (neighbour_size == 1) and (weights[0][0] == active_business):
            return avg
        top = list()
        down = list()
        for item in weights:
            business = item[0]
            key = (active_user, business)
            if business != active_business and key in ratings_data:
                try:
                    value = item[1]
                except:
                    value = 0
                val = value * (ratings_data[(active_user, business)])
                top.append(val)
                down.append(abs(value))
        if sum(down) == 0 or sum(top) == 0:
            return avg
        prediction = float(sum(top) / sum(down))
        pred_abs = abs(prediction)
        return pred_abs

    def find_pearson_corelation(active_business, other_business, users):
        if len(users) == 0:
            return -3.0
        rating1 = [ratings_data[(i, active_business)] for i in users]
        rating2 = [ratings_data[(i, other_business)] for i in users]
        sum1 = sum(rating1)
        sum2 = sum(rating2)
        length1 = len(rating1)
        length2 = len(rating2)
        if length1 != 0 and length2 != 0:
            avg1 = sum1 / length1
            avg2 = sum2 / length2
            diff1 = [k - avg1 for k in rating1]
            diff2 = [k - avg2 for k in rating2]
            if len(diff1) != 0 and len(diff2) != 0:
                root1 = pow(sum([j ** 2 for j in diff1]), 0.5)
                root2 = pow(sum([j ** 2 for j in diff2]), 0.5)
                numerator = sum([diff1[l] * diff2[l] for l in range(min(len(rating1), len(rating2)))])
                denominator = (root1 * root2)
                if denominator != 0 and numerator != 0:
                    return numerator / denominator
                elif denominator == 0 or numerator == 0:
                    return -3.0
        return -3.0

    def find_similarity(active_user, active_business):
        business_similarity = []
        try:
            activeUserBusinesses = u2b_map[active_user]
        except:
            business_similarity.append((active_business, 1.0))
            return business_similarity
        if active_business not in b2u_map:
            business_similarity.append((active_business, 1.0))
            return business_similarity
        try:
            active_user_list = b2u_map[active_business]
        except:
            return None
        if len(active_user_list) == 0:
            return None
        for business in activeUserBusinesses:
            if active_business != business:
                other_business_users = b2u_map[business]
                corrated_users = list(set(active_user_list).intersection(other_business_users))
                if len(corrated_users) != 0:
                    sim = find_pearson_corelation(active_business, business, corrated_users)
                    if sim != -3.0:
                        business_similarity.append((business, sim))
                    else:
                        business_similarity.append((business, 2.5))
                else:
                    business_similarity.append((business, 2.5))
        sim_businesses = sorted(business_similarity, reverse=True)
        return business_similarity

    def flatten_data(item):
        for data_row in item:
            if hasattr(data_row, '__iter__') and not isinstance(data_row, str) and not isinstance(data_row, list):
                for i in flatten_data(data_row):
                    yield i
            else:
                yield data_row

    similar_users = test_data_map.map(lambda q: ((int(q[0]), int(q[1])), find_similarity(q[0], q[1]))).filter(lambda q: q[1] is not None)
    flatten_users = similar_users.map(lambda line: tuple(flatten_data(line)))
    predicted_results = flatten_users.map(lambda q: ((q[0], q[1]), predict_result(q[0], q[1], q[2]))).distinct()
    flattened_pred = predicted_results.map(lambda line: tuple(flatten_data(line)))
    final_result = flattened_pred.map(lambda q: (user_idx_dict[q[0]], bus_idx_dict[q[1]], q[2])).collect()
    result = open(output_file, "w")
    result.write('user_id, business_id, prediction\n')
    for item in final_result:
        result.write(str(item[0]) + "," + str(item[1]) + "," + str(item[2]) + "\n")
    result.close()
    print("Item-Based Time: ", time.time() - start_time)
    sc.stop()


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    case_number = int(sys.argv[3])
    output_file = sys.argv[4]
    if case_number == 1:
        model_based_CF(train_file, test_file, output_file)
    elif case_number == 2:
        user_based_CF(train_file, test_file, output_file)
    elif case_number == 3:
        item_based_CF(train_file, test_file, output_file)