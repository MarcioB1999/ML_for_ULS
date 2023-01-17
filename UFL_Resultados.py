import Classes as Cls
import pickle
import numpy as np
import pandas as pd


def resultados(datafile,n,modelos,indice):

  a=[]
  
  for i in range(n):
      

      Modelo_pi = Cls.ulsr(datafile)
      
      mdl = modelos.modelos[indice]
      mdl.fit(modelos.x_transformed,modelos.y_transformed.ravel())


      y_demanda=modelos.y_scale.inverse_transform(np.array(mdl.predict(modelos.x_scale.fit_transform([[j] for j in range(52)]))).reshape(52,1))

      Modelo_pi.Construir(y_demanda)
      Modelo_pi.Otimizar()
      a.append(Modelo_pi.Resultados())


  return a


with open("modelos_binarios", "rb") as arquivo_binario:
  while True:
    try:
        modelos = pickle.load(arquivo_binario)
    except EOFError:
        break


datafile = "UFL_instancia.txt"

resultados_dataframe_1 = pd.DataFrame(resultados(datafile,200,modelos,0))
resultados_dataframe_2 = pd.DataFrame(resultados(datafile,200,modelos,1))


resultados_dataframe_1.to_csv('Resultados_PI_RR')
resultados_dataframe_1.to_csv('Resultados_PI_RN')