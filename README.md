# Locality-Sensitive-Hashing-based-Sketches
This study delves into the realm of autonomous vehicle tracking utilizing Locality-Sensitive Hashing (LSH), an innovative approach in computer vision and tracking technology. The primary objective is to develop a robust and efficient vehicle tracking system capable of real-time performance in diverse environments.


Traditional object tracking in computer vision often relies on sophisticated deep learning models, leveraging complex architectures like Convolutional Neural Networks (CNNs) and recurrent structures for accurate object detection and tracking. These methods excel in precise localization and tracking of objects within video sequences but encounter challenges in scalability and real-time processing due to their computational demands. To combat these demands and challenges, we propose to revolutionize object tracking by introducing a novel paradigm centered around Locality-Sensitive Hashing (LSH) sketches. These sketches offer a different avenue for object tracking, utilizing probabilistic data structures to approximate nearest neighbor count efficiently. Our proposal maintains the deep learning principles for object detection but shifts the focus from building a solely intricate deep learning architecture to tailored parameter-adjusted LSH sketches. We aim to streamline the tracking process, potentially enabling faster, scalable, and autonomous object tracking in dynamic visual environments. This transition from traditional deep learning-based tracking to our autonomous tracking using locality-sensitive sketches signifies a paradigm shift towards more efficient and adaptable methodologies in computer vision applications.

** Experimental Settings**
Dataset 

For the implementation, we intend on using the "Highway Traffic Videos Dataset" available on Kaggle. This dataset comprises a database of video of traffic on the highway used for monitoring the traffic while also helping to keep track of what kinds of vehicles are traversing through the road. The video was taken over two days from a stationary camera overlooking I-5 in Seattle, WA. The videos were labeled manually as light, medium, and heavy traffic, which correspond respectively to free-flowing traffic, traffic at reduced speed, and stopped or very slow-speed traffic. The video is provided courtesy of the Washington State Department of Transportation [http://www.wsdot.wa.gov/].

From the dataset, we can also understand traffic dynamics, categorizing traffic flow into distinct levels: light, medium, and heavy traffic. These annotations, denoting varying traffic speeds and congestion levels, offer a comprehensive understanding of real-world traffic scenarios. The dataset's stationary camera captures highway footage, providing a valuable resource for analyzing vehicle movement patterns, traffic density, and congestion behaviors. 

In our implementation, we classify the vehicles into ['SUV', 'Sedan', 'Microbus', 'Minivan', 'Truck', 'Bus'].

3.2 Understanding Random Projections for Locality Sensitive Hashing
Our primary aim for using random projection is to encode the high-dimensional feature vectors obtained from the Inception model into lower-dimensional binary representations. For this, we need to create hyperplanes that serve as splitting planes that categorize data points by assigning a value of 0 or 1 based on their position relative to the plane.



Encoding Process:

Splitting Data: The hyperplanes segregate the feature space, assigning 0 to vectors on one side and 1 to vectors on the other side of the hyperplane.

Dot Product Analysis: By utilizing the normal vector of the hyperplane and the vector representations of the data points, a dot product function determines their spatial relationship.
Positive dot product: Indicates that the vector lies on the positive side of the hyperplane.
Negative dot product: Indicates that the vector lies on the negative side.
Zero dot product: Occurs when vectors are perfectly perpendicular to the hyperplane, classifying them as negative direction vectors.



Where our hyperplane normal vector produces a +ve dot-product with another vector, we can view that vector as being in front of the hyperplane. The reverse is true for vectors that produce a -ve dot-product.

Binary Vector Generation:


Information Accumulation: While a single binary value doesn’t provide substantial information about vector similarity, employing multiple hyperplanes amplifies encoded information.

Increased Positional Information: By introducing more hyperplanes, the encoded binary vectors store richer positional information about the original feature vectors.
Parameterization: Choosing the number of hyperplanes (controlled by the 'n bits' parameter) defines the complexity and encoding capacity of the resultant binary vectors.


Enhanced Hashing: Utilizing this concept post the Inception model, the obtained high-dimensional vectors undergo this hashing process to transform them into lower-dimensional binary representations.

A zero shows that the vector is behind the plane (-ve dot product), and a one shows that the vector is in front of the plane (+ve dot product). We combine these to create our binary vectors.

Hence, employing this approach aids in creating a locality-sensitive sketch, enabling efficient and unsupervised object tracking by effectively discerning similarities among object representations. By employing random projection for hashing, the process condenses feature vectors while preserving their distinctive characteristics, paving the way for optimized storage, computational efficiency, and enhanced similarity measurements essential for robust object tracking using unsupervised methodologies. 

The following code snippet gives an idea of how our LSH based sketch works using random projections. 

Let us consider an example to better understand, let the vector input size be 1024, let the sim(essentially our K choices) be 8 and let the nbits be 4. In this case our hashcode/hashsignature would be of length 2048(basically double our input vector size). Moreover the maximum bucket size would be 2^nbits(all possible binary combinations).


**Results and Inferences**

![Table_distribution](https://github.com/Jeffrey-Joan/Locality-Sensitive-Hashing-based-Sketches/assets/57098615/28b31bc8-86a0-4992-b7c0-34ba80e22320)


The above are the distribution plots of each table within a LSH-Sketch. The various subplot corresponds to different hyperparameter(dimension/K choices and nbits) values. We can observe that the plots are promising when dim=32, 16 and when nbits = 2,4.

Now, let us observe the distribution of each hashes over the buckets for dim=32, 16 and when nbits = 2,4.

![hash_distribution](https://github.com/Jeffrey-Joan/Locality-Sensitive-Hashing-based-Sketches/assets/57098615/9a88e6e6-56e4-4f56-affb-6c966b946b20)


There isn’t a strong overlap of the hashes and they have a more variance among each plot/distribution. Which is desirable.

Finally, Let us see how accurate those LSH-sketch with those hyperparameter values are compared to the actual count and the min, median and max of the lookup function for those inputs.

The actual count of each vehicle types are,
Sedan       5776
SUV         1372
Microbus     860
Truck        820
Bus          555
Minivan      467
The actual count of each vehicle types are,
For dim = 32 and nbits = 2,
Total time taken for insertion 320.0221793651581

Total time taken for querying: 0.1980760097503662 

Label:  tensor([0, 0, 0, 0, 1, 0], device='cuda:0')
Max: 7933  Min: 247  Median: 4269.5 
Label:  tensor([1, 0, 0, 0, 0, 0], device='cuda:0')
Max: 7078  Min: 488  Median: 3916.0 
Label:  tensor([1, 0, 0, 0, 0, 0], device='cuda:0')
Max: 7078  Min: 387  Median: 4283.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 7933  Min: 337  Median: 4269.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 7933  Min: 334  Median: 4867.5 
Label:  tensor([1, 0, 0, 0, 0, 0], device='cuda:0')
Max: 6984  Min: 695  Median: 4071.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 7078  Min: 369  Median: 4023.5 
Label:  tensor([0, 0, 0, 0, 1, 0], device='cuda:0')
Max: 7933  Min: 265  Median: 3610.5 

For dim = 32 and nbits = 4,
Total time taken for insertion 266.95368933677673

Total time taken for querying: 0.18964457511901855 

Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 3983  Min: 114  Median: 1419.0 

Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 3983  Min: 157  Median: 1221.0 
Label:  tensor([0, 0, 0, 0, 1, 0], device='cuda:0')
Max: 4412  Min: 618  Median: 1932.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 2994  Min: 175  Median: 1068.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 4412  Min: 417  Median: 1953.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 2994  Min: 70  Median: 1386.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 3983  Min: 244  Median: 1143.0 
Label:  tensor([1, 0, 0, 0, 0, 0], device='cuda:0')
Max: 4412  Min: 244  Median: 1218.0 

For dim = 16 and nbits = 2,
Total time taken for insertion 297.4477939605713

Total time taken for querying: 0.23112082481384277 

Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 8009  Min: 195  Median: 5415.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 7960  Min: 566  Median: 5115.0 
Label:  tensor([0, 0, 0, 0, 1, 0], device='cuda:0')
Max: 8009  Min: 199  Median: 4893.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 7960  Min: 275  Median: 5136.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 8009  Min: 1337  Median: 5562.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 8009  Min: 1337  Median: 5578.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 7960  Min: 123  Median: 4997.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 7960  Min: 389  Median: 5420.5

For dim = 16 and nbits = 4,
Total time taken for insertion 306.4862766265869

Total time taken for querying: 0.3182826042175293 

Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 5098  Min: 138  Median: 1508.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 5377  Min: 214  Median: 1617.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 5377  Min: 95  Median: 1678.5 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 5377  Min: 197  Median: 1399.0 
Label:  tensor([0, 0, 1, 0, 0, 0], device='cuda:0')
Max: 5098  Min: 3  Median: 1798.0 
Label:  tensor([0, 1, 0, 0, 0, 0], device='cuda:0')
Max: 5377  Min: 156  Median: 1680.0 
Label:  tensor([0, 0, 0, 0, 1, 0], device='cuda:0')
Max: 5377  Min: 158  Median: 2008.5 
Label:  tensor([0, 0, 0, 1, 0, 0], device='cuda:0')
Max: 5098  Min: 8  Median: 1976.0 

Unfortunately, we can observe that the Min, Max and median values are similar for different labels. This means that our input vector is quite similar for all the labels. We should be getting much better results if we can generate input vectors that are similar within each category and dissimilar between them.

**Conclusion**

Our proposed LSH-based sketch is a very promising method to estimate count of similar vehicles without any Machine learning algorithms. Also it’s an unsupervised method on its own when we use some basic image processing to get vector representation instead of Neural Networks. Furthermore, The model is also much more efficient in memory and compute compared to Deep Neural Networks. The main problem with the LSH-based sketch is the huge number of hyperparameters that must be manually tuned based. This requires domain expertise and plenty of error and trial. This could make the effective implementation of this a hassle.
