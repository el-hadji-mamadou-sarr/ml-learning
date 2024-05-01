import tensorflow as tf

#print(tf.version)
""" string =tf.Variable("string", tf.string)
rank1= tf.Variable([1.2,1.3],tf.int32)
rank2= tf.Variable([[1.2,1.3],[1.5,1.7]],tf.int32)
print(tf.rank(rank2))
print(rank1.shape) """

""" tensor1=tf.ones([1,2,3]) #mean we have 1 list 2 list inside and each have a list have 3 one
print(tensor1)
tensor2=tf.reshape(tensor1,[2,3,1])
print(tensor2)
tensor3=tf.reshape(tensor1,[3,-1]) # -1 tells tensorflow to calculate the second
print(tensor3) """

t=tf.ones([5,5,5,5,])
print(t)
reshape = tf.reshape(t,[125,-1])
print(reshape)