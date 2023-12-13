import numpy as np

class LSH_sketch:
    def __init__(self, nbits, dim, vec_size):
        self.dim = dim
        self.K = dim
        self.nbits = nbits
        self.vec_size = vec_size
        self.hc_size = (vec_size//dim)*nbits
        self.L = self.hc_size//nbits
        self.B = 2**nbits
        
        self.tables = []
        for i in range(self.L):
            self.tables.append({})

        np.random.seed(42)
        self.plane_norms = np.random.rand(self.nbits, self.dim) - .5
        print('Shape of plane norm',self.plane_norms.shape,'\n')   
        print('Length of vector in',self.vec_size,'\n')   
        print('Length of hachcode',self.hc_size,'\n')   
    

    def insert(self, vector):
    
      
      for count,i in enumerate(self.tables):
        a = (np.dot(vector[count*self.dim:(count+1)*self.dim], self.plane_norms.T)>0).astype(int)
        if str(a) not in i.keys():
          self.tables[count][str(a)]=1
        else:
          self.tables[count][str(a)] += 1          

    def lookup(self, vector):

      
      counts = []
      for count,i in enumerate(self.tables):
      
        a = (np.dot(vector[count*self.dim:(count+1)*self.dim], self.plane_norms.T)>0).astype(int)
        if self.tables[count][str(a)]:
          counts.append(self.tables[count][str(a)])
          

      return counts
    
