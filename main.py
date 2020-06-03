import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from collections import defaultdict
from sklearn.cluster import KMeans

seed_num = 777
np.random.seed(seed_num) # seed setting
iteration = 300 # 할당된 군집이 바뀌지 않을 만큼 반복할 횟수


class kmeans_:
    def __init__(self, k, data, iteration):
        self.k = k # number of cluster
        self.data = data
        self.iteration = iteration

    def Centroids(self):  # k개의 centroids 임의 선택 (randomly choose)
        import random
        length_min = min([i[0] for i in self.data])
        length_max = max([i[0] for i in self.data])
        width_min = min([i[1] for i in self.data])
        width_max = max([i[1] for i in self.data])
        centroids = [[random.uniform(length_min, length_max), random.uniform(width_min, width_max)] for i in range(self.k)]
        return centroids

    def Assignment(self, data_points, centers):  # 각 데이터들을 가장 가까운 centroids가 속한 군집에 할당
        assignments = []
        for point in data_points:
            shortest = 1e9
            shortest_index = 0
            for i in range(len(centers)):
                val = self._distance(point, centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            assignments.append(shortest_index)
        return assignments

    def Update(self, data, assignments):  # Assignment 결과를 바탕으로 centroids 새 지정
        new_means = defaultdict(list)
        centers = []
        for assignment, point in zip(assignments, data):
            new_means[assignment].append(point)
        for points in new_means.values():
            centers.append(self._point_avg(points))
        return centers

    def Train(self):  # 위 과정을 갱신된 centroids가 변화가 없어 할당된 군집이 바뀌지 않을 만큼 반복
        k_points = self.Centroids()
        assignments = self.Assignment(self.data, k_points)
        old_assignments = None
        for i in range(self.iteration):
            new_centers = self.Update(self.data, assignments)
            old_assignments = assignments
            assignments = self.Assignment(self.data, new_centers)
        return zip(assignments, self.data)

    def _distance(self, point, center):
        dimensions = len(point)
        _sum = 0
        for dimension in range(dimensions):
            difference_sq = ((point[dimension] - center[dimension]) ** 2)
            _sum += difference_sq
        return sqrt(_sum)

    def _point_avg(self, points):
        dimensions = len(points[0])
        new_center = []
        for dimension in range(dimensions):
            dim_sum = 0
            for p in points:
                dim_sum += p[dimension]
            new_center.append(dim_sum / float(len(points)))
        return new_center


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    my_data = pd.read_csv('data.csv').to_numpy().tolist()

    # kmeans_ 결과 확인
    model1 = kmeans_(k=3, data=my_data, iteration=iteration)
    predict1 = np.array(list(model1.Train()))
    label = predict1[:, 0]
    coords = np.array([[item[0], item[1]] for item in predict1[:, 1]])

    plt.subplot(2, 1, 1)
    plt.scatter(coords[:, 0], coords[:, 1], c=label)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.title("implementaion")

    # scikit-learn의 Kmeans 모듈을 이용해서 결과 검증해보기
    model2 = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data)
    predict = pd.DataFrame(model2.predict(data))
    predict.columns=['predict']
    r = pd.concat([data, predict], axis=1)

    plt.subplot(2, 1, 2)
    plt.scatter(r['Sepal length'], r['Sepal width'], c=r['predict'])
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.title("from scikit-learn library")
    plt.savefig("./result.png")
    plt.show()