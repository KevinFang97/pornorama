import cv2
import numpy as np
from typing import List
from collections import defaultdict
import copy
import matplotlib.pyplot as plt

def extract_sift(image_list:List[np.ndarray]):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
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


def extract_ORB(image_list:List[np.ndarray]):
    orb = cv2.ORB_create()
    kp_list = []
    feature_list = []
    for each_image in image_list:
        if each_image.shape[2] != 1:
            img_gray = cv2.cvtColor(each_image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = each_image

      #  kp = orb.detect(img_gray, None)
        kp, descriptor = orb.detectAndCompute(img_gray, None)
        kp_list.append(kp)
        feature_list.append(descriptor)

    return kp_list, feature_list

'''

return sets of feature matchees if there are multiple panoramas
input: list of feature descriptors

output: 1. set of sub maximum spanning tree with eahc node in each set like
            (img_id1, img_id2, num_feature_matches)

        2. dictionary of matches between images, keys are image ids.
'''
def feature_match(feature_list, threshold=50, SIFT=True):
    match_dic = {}
    if SIFT:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    num_images = len(feature_list)
    graph = defaultdict(dict)
    for i in range(num_images):
        for j in range(i+1, num_images):
            des1 = feature_list[i]
            des2 = feature_list[j]
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            matches = list(filter(lambda x:x.distance < 100, matches))
            match_dic[(i, j)] = matches
            match_dic[(j, i)] = matches

            if len(matches) > threshold:
                graph[i][j] = len(matches)
                graph[j][i] = len(matches)

    #Seprate the graph into sets of sub-graph that are previously not connected using DFS
    sub_graph_set = []
    graph_add = copy.deepcopy(graph)
    key_set = set()
    while len(graph) != 0:
        each_key = list(graph.keys())[0]
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

    return result, match_dic


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
        if farther_node[id] != id:
            return find(farther_node[id])
        else:
            return id

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


if __name__ == '__main__':

	img_path1 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/fishbowl/fishbowl-00.png'
	img_path2 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/fishbowl/fishbowl-01.png'
	img_path3 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/fishbowl/fishbowl-02.png'
	img_path4 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/fishbowl/fishbowl-03.png'
	img_path5 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/halfdome/halfdome-00.png'
	img_path6 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/halfdome/halfdome-01.png'
	img_path7 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/halfdome/halfdome-02.png'
	img_path8 = '/home/han/gitlab/CMU16720/HW2/HW2/data/adobe_panoramas/data/halfdome/halfdome-03.png'

	img_color1 = cv2.imread(img_path1)
	img_color2 = cv2.imread(img_path2)
	img_color3 = cv2.imread(img_path3)
	img_color4 = cv2.imread(img_path4)
	img_color5 = cv2.imread(img_path5)
	img_color6 = cv2.imread(img_path6)
	img_color7 = cv2.imread(img_path7)
	img_color8 = cv2.imread(img_path8)

	#height, width,_ = img_color2.shape
	#img_color2 = cv2.resize(img_color2, (int(width/2), int(height/2)))
	#img_color2 = cv2.transpose(img_color2)
	#img_color2 = cv2.flip(img_color2,flipCode=0)

	# rotate cw
	##out=cv2.transpose(img_color2)
	#out=cv2.flip(out,flipCode=1)

	img_list = [img_color1, img_color2, img_color3, img_color4, img_color5, img_color6, img_color7, img_color8]
	kp_list, feature_list = extract_ORB(img_list)

    '''
	kp1 = kp_list[0]
	kp2 = kp_list[1]

	des1 = feature_list[0]
	des2 = feature_list[1]
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x: x.distance)

	img3 = cv2.drawMatches(img_color1,kp1, img_color2, kp2, matches[0:30], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	plt.imshow(img3)
	plt.show()
    result, match_dic = feature_match(feature_list)
    '''

	'''
	#Visualizing the result
	'''
	for idx, each_result in enumerate(result):
	    img_set = set()
	    for pair in each_result:
		img_set.add(pair[0])
		img_set.add(pair[1])
	    img_id_list = list(img_set)
	    num_images = len(img_id_list)
	    height, width, channel = img_list[img_id_list[0]].shape
	    concatenated = np.zeros([height, width*4, channel])
	    for i in range(num_images):
            concatenated[:, i*width:(i+1)*width,:] = img_list[img_id_list[i]]


	    for i in range(num_images-1):
            idx1 = img_id_list[i]
            idx2 = img_id_list[i+1]
            matches = match_dic[(idx1, idx2)]

		for each_match in matches[0:30]:
		    train_idx = each_match.trainIdx
		    query_idx = each_match.queryIdx
		    pos1 = kp_list[idx1][query_idx].pt
		    pos2 = kp_list[idx2][train_idx].pt
		    pos1 = (int(pos1[0]+i*width), int(pos1[1]))
		    pos2 = (int(pos2[0]+(i+1)*width), int(pos2[1]))
		    cv2.line(concatenated, pos1, pos2, (0, 255, 0), thickness=1, lineType=8)

	    cv2.imwrite('result{}.jpg'.format(idx+1), concatenated)
