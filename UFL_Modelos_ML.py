import Classes as Cls
import pandas as pd
from sklearn.neural_network import MLPRegressor as rede_neural
from sklearn.ensemble import RandomForestRegressor
import pickle

n=10

funcao_demanda = lambda i,j: [(i*6-j*2)**2+10000]

x = [[j] for j in range(52) for i in range(n)]

y = pd.read_csv('Resultados/demandas')['y'].values

modelos = Cls.modelos_regressor([RandomForestRegressor(n_estimators=10),rede_neural()],
                            x,y)

modelos.Transformar()

modelos.Treinar()
modelos.Avaliar()



resultados = pd.DataFrame(
                   {'Random forest': modelos.medias_desempenho[0],
                    'Rede Neural': modelos.medias_desempenho[1],}
                    )

resultados.to_csv('Resultados/Resultados_desempenho')

with open('Resultados/modelos_binarios', 'ab') as arquivo_binario:
    pickle.dump(modelos, arquivo_binario)