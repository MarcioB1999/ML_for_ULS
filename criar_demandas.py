import Classes as Cls
import pandas as pd


funcao_demanda = lambda i,j: [(i*6-j*2)**2+10000]
n = 10

produto = Cls.itens()
produto.Construcao(funcao_demanda, n)

demandas = pd.DataFrame({'y':produto.demandas})
demandas.to_csv('demandas')
