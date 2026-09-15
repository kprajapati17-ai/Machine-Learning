import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Column 1 = Student Marks

marks = np.array([
    [78], [65], [45], [92], [56], [88], [71], [34], [99], [62],
    [53], [81], [47], [74], [68], [91], [39], [85], [59], [76],
    [43], [97], [64], [52], [83], [70], [36], [89], [61], [55],
    [94], [48], [73], [67], [82], [41], [58], [96], [63], [77],
    [35], [86], [69], [54], [93], [72], [49], [87], [60], [79],
    [44], [98], [66], [51], [84], [38], [75], [57], [90], [46],
    [80], [33], [95], [68], [42], [71], [88], [53], [62], [99],
    [37], [76], [59], [81], [65], [47], [92], [56], [73], [85],
    [50], [78], [34], [91], [63], [69], [45], [87], [55], [96],
    [61], [74], [40], [83], [58], [89], [52], [67], [79], [48]
])


# ==========================================
# 2. Student Names
# ==========================================

students = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Krish", "Rudra", "Dhruv",
    "Aryan", "Reyansh", "Kabir", "Ayaan", "Vihaan", "Rohan", "Yash",
    "Harsh", "Dev", "Raj", "Manav", "Kunal", "Parth", "Meet", "Jai",
    "Darsh", "Neel", "Ved", "Shiv", "Moksh", "Tanish", "Om", "Soham",
    "Ayush", "Nirav", "Hiten", "Mihir", "Ansh", "Kartik", "Ronak",
    "Sagar", "Aniket", "Rahul", "Akash", "Nikhil", "Vatsal", "Chirag",
    "Bhavin", "Hardik", "Tirth", "Maulik", "Jigar", "Het", "Darshan",
    "Vivek", "Amit", "Karan", "Ritesh", "Nayan", "Kishan", "Bhargav",
    "Ravi", "Smit", "Akshat", "Manish", "Vishal", "Pratik", "Nirmit",
    "Yuvraj", "Siddharth", "Abhishek", "Rishabh", "Tejas", "Viren",
    "Naitik", "Aayush", "Rajesh", "Sachin", "Piyush", "Anuj", "Hemant",
    "Vijay", "Gaurav", "Mohit", "Varun", "Deep", "Kush", "Rakesh",
    "Samir", "Akhil", "Mayank", "Rajat", "Suraj", "Prem", "Naman",
    "Aman", "Ketan", "Dhiren", "Jayesh", "Darshit", "Ronit", "Sahil",
    "Utsav"
]


model = KMeans(n_clusters=3,random_state=45,n_init=10)
model.fit(marks)

labels =model.labels_
centroids = model.cluster_centers_.flatten()

print("Cluster Centroids:")
print(centroids)

cluster = np.argsort(centroids)

weak_cluster = cluster[0]
average_cluster = cluster[1]
excellent_cluster = cluster[2]

performance_map ={
    weak_cluster : 'weak',
    average_cluster :'average',
    excellent_cluster:'excellent'
}

for name,score,cluster in zip(students,marks.flatten(),labels):
    category = performance_map[cluster]
    print(f"{name} has {score} marks and is an {category} student.")

plt.scatter(marks.flatten(),labels)

plt.xlabel("Student Marks")
plt.ylabel("Cluster")
plt.title("Student Performance Clustering")

plt.show()