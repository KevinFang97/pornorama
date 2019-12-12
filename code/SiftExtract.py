import cv2
import numpy as np
from typing import List
from collections import defaultdict
import copy

def extract_sift(image_list:List[np.ndarray]):
    sift = cv2.xfeatures2d.SIFT_create()
    kp_list = []
    feature_list = []
    for each_image in image_list:
        if each_image.shape[2] != 1:
            img_gray = cv2.cvtColor(each_image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = each_image

        kp,descriptor = sift.detectAndCompute(img_gray, None)
        kp_list.append(kp)
        feature_list.append(descriptor)

    return kp_list, feature_list


'''
return sets of feature matchees if there are multiple panoramas
'''
def feature_match(feature_list, threshold=100):

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    num_images = len(feature_list)
    graph = defaultdict({})

    for i in range(num_images):
        for j in range(i+1, num_images):
            des1 = feature_list[i]
            des2 = feature_list[j]
            matches = bf.match(des1, des2)

            if len(matches) > threshold:
                graph[i][j] = len(matches)
                graph[j][i] = len(matches)

    #Seprate the graph into sets of sub-graph that are previously not connected using DFS
    sub_graph_set = []
    graph_add = copy.deepcopy(graph)
    key_set = set()
    while len(graph) != 0:
        for each_key in graph.keys():
            key_set.add(each_key)
            find_sub_graph(graph, key_set, each_key)
            sub_graph = {}
            for same_set_key in key_set:
                sub_graph[same_set_key] = graph_add[same_set_key]
                del graph[same_set_key]
            sub_graph_set.append(sub_graph)
            key_set.clear()

    #Then use maximum spanning tree over each sub-graph to find strongest connection
    result = []
    for each_graph in sub_graph_set:
        mst = maximum_spanning_tree(each_graph)
        result.append(mst)

    return result


def find_sub_graph(graph, key_set, cur_key):
    node = graph[cur_key]

    for each_key in node.keys():
        if each_key not in key_set:
            key_set.add(each_key)
            find_sub_graph(graph, key_set, each_key)


def maximum_spanning_tree(graph):
    farther_node = {}
    def union(id1, id2):

        if find(id1) == find(id2):
            return False
        else:
            farther_node[id2] = farther_node[id1]
            return True

    def find(id):
        while farther_node[id] != id:
            return find(farther_node[id])

    connection_list = []
    added_set = set()
    for each_node_id in graph.keys():
        farther_node[each_node_id] = each_node_id
        for connection_node_id in graph[each_node_id].keys():
            if (each_node_id, connection_node_id) not in added_set \
                    and (connection_node_id, each_node_id) not in added_set:
                added_set.add((each_node_id, connection_node_id))
                added_set.add((connection_node_id, each_node_id))
                connection_list.append([each_node_id, connection_node_id, graph[each_node_id][connection_node_id]])

    #Sort the matched feature points number from large to small
    connection_list = sorted(connection_list, key=lambda x: x[2], reverse=True)
    result = []

    for each_connection in connection_list:
        if union(each_connection[0], each_connection[1]):
            result.append(each_connection)
            continue
        else:
            break

    return result



img_path1 = '../data/incline_L.png'
img_path2 = '../data/incline_R.png'

img_color1 = cv2.imread(img_path1)
img_color2 = cv2.imread(img_path2)
img_list = [img_color1, img_color2]
extract_sift(img_list)