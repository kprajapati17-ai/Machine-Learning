def GetMean(data):
    total = 0

    for i in data:
        total = total + i

    avg = total/len(data)

    return avg

def Getvariance(data,avg):
    deviation =[]

    for item in data:
        diff= item-avg
        deviation.append(diff)

    square =[]
    total=0

    for item in deviation:
        sqr= item * item
        total = total +sqr
        square.append(sqr)

    population_variance = total / len(data)

    population_variance = round(population_variance,2)

    simple_variance = total/(len(data)-1)

    simple_variance = round(simple_variance,2)

    return population_variance,simple_variance

def getCovariance(data1, data2):
    m1 = GetMean(data1)
    m2 = GetMean(data2)
    total = 0

    for i in range(len(data1)):
        deviation1= data1[i]-m1
        deviation2= data1[i]-m2

        total = total +(deviation1 * deviation2)


    
    population_variance = total / len(data1)
    
    population_variance = round(population_variance,2)
    
    simple_variance = total/(len(data1)-1)
    
    simple_variance = round(simple_variance,2)
    
    return population_variance,simple_variance