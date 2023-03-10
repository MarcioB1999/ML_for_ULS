import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler


class ulsr:


  def __init__(self,datafile):
    self.datafile = datafile
    self.modelo = None
    self.demandas = None
    self.x = None
    self.s = None
    self.y = None
    self.xr = None
    self.sr = None
    self.yr = None
    self.ObjVal = None


  def Construir(self,predictD):
    

    with open(self.datafile, 'r') as file: linhas = file.readlines()

    # remove linha vazia inicial e elimina os "\n" de cada linha
    linhas = [a.strip() for a in linhas] 


    # ler o tamanho da instancia
    
    N = int(linhas[0])
  #  N = 5 
      
    H = np.zeros(N)#custo no estoque
    P = np.zeros(N)#custo unitário
    F = np.zeros(N)#custo fixo
    HR = np.zeros(N)
    PR = np.zeros(N)
    FR = np.zeros(N)
    D = np.zeros(N)#demanda
    R = np.zeros(N)#remanufatura
    

    FR = [float(linhas[1]) for i in range(N)]
    #print('FR',FR)



    F = [float(linhas[2]) for i in range(N)]
    #print('F',F)


    HR = [float(linhas[3]) for i in range(N)]
    #print('HR',HR)


    H = [float(linhas[4]) for i in range(N)]
    #print('H',H)

    #D = np.fromstring(linhas[5], dtype=float, sep = ' ')
    D = predictD
    #print(D)
    
  

    R = D/100#np.fromstring(linhas[6], dtype=float, sep = ' ')
    #print('R',R)
    

    #cria o modelo
    m = gp.Model("ulsr") 
    m.Params.LogToConsole = 0

    #Adicionando Variáveis
    x = m.addVars(N, name='x') 
    s = m.addVars(N, name='s')  
    y = m.addVars(N, vtype=GRB.BINARY, name='y') 
    xr = m.addVars(N, name='xr') 
    sr = m.addVars(N, name='sr')  
    yr = m.addVars(N, vtype=GRB.BINARY, name='yr') 

    # funcao objetivo
    obj = 0
    for i in range(0, N):
      obj += P[i] * x[i]
      obj += H[i] * s[i]
      obj += F[i] * y[i]
      obj += PR[i] * xr[i]
      obj += HR[i] * sr[i]
      obj += FR[i] * yr[i]

    m.setObjective(obj, GRB.MINIMIZE)
    
    m.addConstr(x[0] + xr[0] - s[0] == D[0])
    for i in range(1, N):
      m.addConstr(s[i-1] + x[i] + xr[i] - s[i] == D[i])

    m.addConstr(- xr[0] - sr[0] == - R[0])
    for i in range(1, N):
      m.addConstr(sr[i-1] - xr[i] - sr[i] == -R[i])
    
    
    for i in range(0, N):
      m.addConstr(x[i] - (D[i:N].sum())*y[i] <= 0)

    for i in range(0, N):
      m.addConstr(xr[i] - min((D[i:N].sum()), (R[0:i+1].sum()))*yr[i] <= 0 )
      
  #  for i in range(0, N):
  #    m.addConstr(xr[i] - D[i:N].sum()*yr[i] <= 0 )

    m.addConstr(s[N-1] == 0)

    self.modelo = m
    self.demandas = D
    self.x = x
    self.s = s
    self.y = y
    self.xr = xr
    self.sr = sr
    self.yr = yr




  def Otimizar(self):

    self.modelo.optimize()
    self.ObjVal = self.modelo.ObjVal
  

  def Resultados(self):
    resultados = {'x': [self.x[i].getAttr("x") for i in self.x],
    's':  [self.s[i].getAttr("x") for i in self.s],
    'y': [self.y[i].getAttr("x") for i in self.y],
    'xr': [self.xr[i].getAttr("x") for i in self.xr], 
    'sr': [self.sr[i].getAttr("x") for i in self.sr],
    'yr': [self.yr[i].getAttr("x") for i in self.yr],
    'ObjVal': self.modelo.ObjVal
    }

    return resultados





class modelos_regressor:
    #oq tem q fazer?
    #treinar, r^2, predizer e avaliar o desempenho estatisticamente
    #oq precisa ter?
    #r^2, vetor de resultados das medias
    def __init__(self,modelos,x,y):
      self.modelos = modelos
      self.medias_desempenho = False
        
      self.y=y
      self.y_scale = False
      self.y_transformed = False

      self.x = x
      self.x_scale = False
      self.x_transformed = False



    def Transformar(self):
      self.y_scale = StandardScaler()
      y_transformed = self.y_scale.fit_transform(np.array(self.y).reshape(-1, 1))

      self.x_scale = StandardScaler()
      x_transformed = self.x_scale.fit_transform(self.x)
        
      self.y_transformed = y_transformed
      self.x_transformed = x_transformed




    def Treinar(self):
      for modelo in self.modelos:
        modelo.fit(self.x_transformed,self.y_transformed.ravel())
    


    def Avaliar(self):
      medias_modelos = {}
      indice_modelo = 0
      for modelo in self.modelos:


        medias_modelo = []
       
        for i in range(120):
          
          kfold = KFold(n_splits=10, shuffle=True, random_state=i)

          model = modelo
          scores = cross_val_score(model, self.x_transformed, self.y_transformed.ravel(), cv = kfold)
          medias_modelo.append(scores.mean())

        medias_modelos[indice_modelo] = medias_modelo
        indice_modelo += 1

      self.medias_desempenho = medias_modelos


    
    def Predizer(self,indice,X):
      X = self.x_scale.fit_transform(X)
      return self.modelos[indice].predict(X)






class itens:

    def __init__(self):
        self.demandas = []
        self.medias = []
        

    def Construcao(self,funcao,amostra_periodo):

      medias_aux = [funcao(i,j) for i in range(13) for j in range(1,5)]
  
      y_item = []
      for media_periodo in medias_aux:
        y_item = y_item + list(np.random.poisson(media_periodo, amostra_periodo))
            
        
      self.medias = medias_aux
      self.demandas = y_item
