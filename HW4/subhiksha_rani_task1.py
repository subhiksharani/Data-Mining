import sys
from pyspark import SparkContext
import time
from operator import add
import itertools
from queue import *

sc = SparkContext(appName="task1")


def generate_pairs(user):
    bus_id = user[0]
    uid = user[1]
    output = []
    for i in itertools.combinations(uid, 2):
        i = sorted(i)
        output.append(((i[0], i[1]), [bus_id]))
    return output


def find_adjacent_vertices(edge, edge_list):
    output = []
    for i in edge_list:
        if i[0] == edge:
            output.append(i[1])
        elif i[1] == edge:
            output.append(i[0])
    output = list(set(output))
    return output


def compute_betweenness(root_node, adjacent_vertices, v_len):
    visited = []
    levels = {}
    parents = {}
    weights = {}
    que = Queue(maxsize=v_len)
    que.put(root_node)
    visited.append(root_node)
    levels[root_node] = 0
    weights[root_node] = 1
    while que.empty() is not True:
        node = que.get()
        children = adjacent_vertices[node]
        for i in children:
            if i not in visited:
                que.put(i)
                parents[i] = [node]
                weights[i] = weights[node]
                visited.append(i)
                levels[i] = levels[node] + 1
            else:
                if i != root_node:
                    parents[i].append(node)
                    if levels[node] == levels[i] - 1:
                        weights[i] += weights[node]
    order = []
    count = 0
    for i in visited:
        order.append((i, count))
        count = count + 1
    sort_order = sorted(order, key=(lambda x: x[1]), reverse=True)
    reverse = []
    node_values = {}
    betweenness = {}
    for i in sort_order:
        reverse.append(i[0])
        node_values[i[0]] = 1
    for j in reverse:
        if j != root_node:
            weight = 0
            for i in parents[j]:
                if levels[i] == levels[j] - 1:
                    weight += weights[i]
            for i in parents[j]:
                if levels[i] == levels[j] - 1:
                    source = j
                    destination = i
                    if source < destination:
                        pair = tuple((source, destination))
                    else:
                        pair = tuple((destination, source))
                    if pair not in betweenness.keys():
                        betweenness[pair] = float(node_values[source] * weights[destination] / weight)
                    else:
                        betweenness[pair] = betweenness[pair] + float(node_values[source] * weights[destination] / weight)
                    node_values[destination] = node_values[destination] + float(node_values[source] * weights[destination] / weight)
    result = []
    for k, v in betweenness.items():
        value = [k, v]
        result.append(value)
    return result


def check_if_empty(vertices):
    if len(vertices) == 0:
        return True
    else:
        for i in vertices.keys():
            adjacency_list = vertices[i]
            if len(adjacency_list) != 0:
                return False
            else:
                pass
        return True


def bfs(root, adjacent_matrix, vertex_len):
    vertices = []
    edges = []
    que = Queue(maxsize=vertex_len)
    que.put(root)
    vertices.append(root)
    while que.empty() is not True:
        node = que.get()
        children = adjacent_matrix[node]
        for i in children:
            if i not in vertices:
                que.put(i)
                vertices.append(i)
            pair = sorted((node, i))
            if pair not in edges:
                edges.append(pair)
    return (vertices, edges)


def delete_component(rem_graph, value):
    component_vertices = value[0]
    for v in component_vertices:
        del rem_graph[v]
    for i in rem_graph.keys():
        adjacency_list = rem_graph[i]
        for v in component_vertices:
            if v in adjacency_list:
                adjacency_list.remove(i[1])
        rem_graph[i] = adjacency_list
    return rem_graph


def find_linked_edges(adjacency_matrix):
    result = []
    rem_graph = adjacency_matrix
    while check_if_empty(rem_graph) is False:
        vertices = []
        for k, v in rem_graph.items():
            vertices.append(k)
        vertices = list(set(vertices))
        root = vertices[0]
        bfs_result = bfs(root, adjacency_matrix, len(vertices))
        result.append(bfs_result)
        rem_graph = delete_component(rem_graph, bfs_result)
    return result


def compute_modularity(adjacent_vertices, connected_edges, edge_len):
    mod = 0
    for c in connected_edges:
        vertices = c[0]
        for i in vertices:
            for j in vertices:
                a_ij = 0
                adjacency_list = adjacent_vertices[str(i)]
                if j in adjacency_list:
                    a_ij = 1
                ki = len(adjacent_vertices[i])
                kj = len(adjacent_vertices[j])
                mod += a_ij - (ki * kj) / (2 * edge_len)
    mod = mod / (2 * edge_len)
    return mod


def delete_edge(adjacency_matrix, delete_edge):
    if delete_edge[0] in adjacency_matrix.keys():
        edges = adjacency_matrix[delete_edge[0]]
        if delete_edge[1] in edges:
            edges.remove(delete_edge[1])
    if delete_edge[1] in adjacency_matrix.keys():
        edges = adjacency_matrix[delete_edge[1]]
        if delete_edge[0] in edges:
            edges.remove(delete_edge[0])
    return adjacency_matrix


def construct_adjacency_matrix(components):
    result = {}
    for item in components:
        edges = item[1]
        for i in edges:
            if i[0] in result.keys():
                result[i[0]].append(i[1])
            else:
                result[i[0]] = [i[1]]
            if i[1] in result.keys():
                result[i[1]].append(i[0])
            else:
                result[i[1]] = [i[0]]
    return result


def write_to_file(sort_betweenness, sort_communities, out_file1, out_file2):
    file1 = open(out_file1, 'w')
    count1 = 0
    betweenness_list = sort_betweenness.collect()
    for i in betweenness_list:
        if count1 == 0:
            count1 = 1
        else:
            file1.write("\n")
        file1.write(str(i[0]) + ", " + str(i[1]))
    file1.close()
    file2 = open(out_file2, 'w')
    count2 = 0
    for i in sort_communities:
        if count2 == 0:
            count2 = 1
        else:
            file2.write("\n")
        s = str(i[0]).replace("[", "").replace("]", "")
        file2.write(s)
    file2.close()


def task1(input_file, filter_threshold, output1, output2):
    data = sc.textFile(input_file)
    full_data = data.filter(lambda line: line[0] != "user_id").map(lambda line: line.split(',')).persist()
    bus_users_map = full_data.map(lambda q: (q[1], [q[0]])).reduceByKey(add)
    user_pairs = bus_users_map.flatMap(lambda q: generate_pairs(q)).reduceByKey(add)
    filter_users = user_pairs.filter(lambda q: len(q[1]) >= filter_threshold).map(lambda x: x[0])
    vertices = filter_users.flatMap(lambda q: [(q[0]), (q[1])]).distinct()
    edges = filter_users.map(lambda q: (q[0], q[1])).collect()
    adjacent_vertices = vertices.map(lambda q: (q, find_adjacent_vertices(q, edges))).collectAsMap()
    v_len = vertices.count()
    betweenness = vertices.flatMap(lambda q: compute_betweenness(q, adjacent_vertices, v_len)).reduceByKey(add)
    sort_betweenness1 = betweenness.map(lambda q: (q[0], float(q[1] / 2)))
    sort_betweenness = sort_betweenness1.map(lambda q: (q[1], q[0])).sortByKey(ascending=False).map(lambda q: (q[1], q[0]))
    del_edge = sort_betweenness.take(1)[0][0]
    edge_len = len(edges)
    adj_matrix = adjacent_vertices.copy()
    highest_mod = -1
    communities = []
    count = 0
    while count < 50:
        adj_matrix = delete_edge(adj_matrix, del_edge)
        linked_edges = find_linked_edges(adj_matrix)
        mod = compute_modularity(adjacent_vertices, linked_edges, edge_len)
        adj_matrix = construct_adjacency_matrix(linked_edges)
        value = []
        for i in adj_matrix.keys():
            value.append(i)
        value = list(set(value))
        all_values = sc.parallelize(value)
        betweenness = all_values.flatMap(lambda x: compute_betweenness(x, adj_matrix, v_len)).reduceByKey(add)
        sorted_bet = betweenness.map(lambda x: (x[0], float(x[1] / 2))).map(lambda x: (x[1], x[0])).sortByKey(ascending=False).map(lambda x: (x[1], x[0]))
        del_edge = sorted_bet.take(1)[0][0]
        if mod >= highest_mod:
            highest_mod = mod
            communities = linked_edges
        count = count + 1
    sort_communities = []
    for i in communities:
        item = sorted(i[0])
        sort_communities.append((item, len(item)))
    sort_communities.sort()
    sort_communities.sort(key=lambda x: x[1])
    write_to_file(sort_betweenness, sort_communities, output1, output2)


if __name__ == '__main__':
    input_file = sys.argv[2]
    output_file1 = sys.argv[3]
    output_file2 = sys.argv[4]
    filter_threshold = int(sys.argv[1])
    start_time = time.time()
    task1(input_file, filter_threshold, output_file1, output_file2)
    end_time = time.time()
    print("Total time: ", end_time - start_time)
    sc.stop()