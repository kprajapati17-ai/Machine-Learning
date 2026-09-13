
import Mymath as m

weight = [55, 65, 72, 50, 80, 68, 58, 75, 53, 85]
w_mean = m.GetMean(weight)
print("Weight mean:",w_mean)
W_result = m.Getvariance(weight,w_mean)
print(W_result)


height = [165.5, 170.2, 175.8, 160.4, 180.1,
          172.6, 168.3, 177.5, 163.7, 182.9]
h_mean = m.GetMean(height)
print("Height mean:",h_mean)
H_result = m.Getvariance(height,h_mean)
print(H_result)

covariance_result = m.getCovariance(height,weight)

print("Population Covariance:", covariance_result[0])
print("Sample Covariance:", covariance_result[1])