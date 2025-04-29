#input data：data_ is source_data，data_0 is "label = 0" data，data_1 is "label = 1"data,data_2 is"label = 2"data
data_ =  np.loadtxt('./Data_.txt')
data_0 = np.loadtxt('./Data_0.txt')
data_1 = np.loadtxt('./Data_1.txt')
data_2 = np.loadtxt('./Data_2.txt')

data_apply_global = np.loadtxt('./global_dataset.txt')


#Disarrange the sample order of the data set
np.random.shuffle(data_)
np.random.shuffle(data_0)
np.random.shuffle(data_1)
np.random.shuffle(data_2)


#DataMode：Stratified sampling was conducted directly (7:3)
x1 = data_0[0:int(data_0.shape[0]*0.7)]
x2 = data_1[0:int(data_1.shape[0]*0.7)]
x3 = data_2[0:int(data_2.shape[0]*0.7)]
Data_Train = np.vstack((x1,x2,x3))

x1 = data_0[int(data_0.shape[0]*0.7):int(data_0.shape[0])]
x2 = data_1[int(data_1.shape[0]*0.7):int(data_1.shape[0])]
x3 = data_2[int(data_2.shape[0]*0.7):int(data_2.shape[0])]
Data_Test = np.vstack((x1,x2,x3))

Data_Train_X = Data_Train[:,0:10]
Data_Train_Y = Data_Train[:,11]
Data_Test_X = Data_Test[:,0:10]
Data_Test_Y = Data_Test[:,11]