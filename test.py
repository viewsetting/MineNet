import Networks as mn
import  numpy as np
import  matplotlib.pyplot as plt
#n =mn.NeuralNetwork()

# file = open("mnist_train.csv","r")
# data_lst = file.readlines()
# file.close()

#PARAMETERS OF NEURAL NETWORKS
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.15

#THIS IS NN
n = mn.NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#LOAD TRAIN DATA
train_file = open("mnist_train.csv","r")
train_lst = train_file.readlines()
train_file.close()
print("DATA HAS BEN LOADED.")

epochs = 5
data_size= len(train_lst)
print(data_size)
show_intervine=3000
for t in range(epochs):
    step = 1
    for num in train_lst:
        all_value = num.split(',')

        #input list
        inputs = (np.asfarray(all_value[1:])/255.0*0.99) + 0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_value[0])]=0.99  #all_value[0] is the true number value
        n.train(inputs,targets)

        if step % show_intervine == 1 :
            print("epochs :",t," rate :",step*100.0/data_size,"%")

        step += 1
        pass

    pass

#LOAD TEST DATA
test_file = open("mnist_test.csv","r")
test_lst = test_file.readlines()
test_file.close()

rec = []
step=0
show_intervine=123
score=0
for record in test_lst:
    step += 1
    all_value = record.split(',')
    answer = all_value[0]
    inputs = (np.asfarray(all_value[1:])/255.0*0.99) + 0.01
    outputs= n.query(inputs)
    arg_op = np.argmax(outputs)
    # if step % show_intervine == 1 :
    #     print("This IMAGE is : ",answer,"  MINENET's answer : ",arg_op," outputs :",outputs)
    #     image = np.asfarray(all_value[1:]).reshape(28,28)
    #     plt.imshow(image,cmap='Greys',interpolation='None')
    #     plt.show()

    if int(answer) == int(arg_op):
        score = score+1
    else:
       pass

rec_a=np.asarray(rec)

print("Score: ",score," Sum: ",len(test_lst))
