# Laboratório de Modelos de Machine Learning e Deep Learning para Criptomoedas

Este repositório contém um sistema que funciona como um laboratório para aplicar modelos de Machine Learning (ML), Deep Learning (DL) e Aprendizado por Reforço (RL) em dados de criptomoedas. O objetivo principal é realizar análises, backtests e otimizações em estratégias de negociação utilizando dados históricos refinados, que são obtidos de um Data Lake previamente alimentado por um sistema de ingestão chamado `ingest_crypto-data`.

## Estrutura do Projeto

Abaixo está a descrição detalhada de cada arquivo presente no projeto, incluindo os que estão sendo utilizados na estratégia principal (`estrategia1`) e os que estão desabilitados.

---

### **1. `main.py`**
- **Descrição**: Arquivo principal que inicia a execução do sistema. Ele chama a estratégia `estrategia1` e executa o backtest.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Inicializa a estratégia `estrategia1`.
  - Executa o backtest otimizado.

---

### **2. `estrategia1.py`**
- **Descrição**: Implementa a estratégia principal do sistema. É responsável por carregar os dados, configurar o backtester e executar o pipeline completo de treinamento e avaliação dos modelos.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Carrega os dados históricos de um arquivo Parquet.
  - Executa o backtest otimizado utilizando o `Backtester`.
  - Insere métricas de desempenho no banco de dados.

---

### **3. `backtester.py`**
- **Descrição**: Classe que realiza o backtest das estratégias. Integra os modelos de ML, DL e RL para avaliar o desempenho em dados históricos.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Treina e avalia os modelos (`XGBoost`, `LSTM`, `MLP`, `Stacking Ensemble` e `PPO`).
  - Realiza o backtest em janelas temporais de 30 dias.
  - Retorna métricas detalhadas de desempenho.

---

### **4. `performance_models.py`**
- **Descrição**: Classe que calcula indicadores de desempenho e gera relatórios em PDF com gráficos e tabelas.
- **Status**: **Não utilizado na estratégia atual**.
- **Funcionalidade**:
  - Calcula métricas como taxa de sucesso, drawdown máximo, ganhos/perdas consecutivas, etc.
  - Gera gráficos de retorno acumulado e valor de investimento semanal.
  - Cria relatórios em PDF e faz upload para o S3.

---

### **5. `data_handler.py`**
- **Descrição**: Classe responsável por manipular os dados, incluindo o pré-processamento e o upload de datasets para o S3.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Carrega dados históricos do Data Lake.
  - Remove colunas irrelevantes e escala os dados.
  - Salva datasets e parâmetros de escalonamento no S3.

---

### **6. `mlp_classification.py`**
- **Descrição**: Implementa um modelo de classificação utilizando uma Rede Neural Perceptron Multicamadas (MLP).
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Treina o modelo MLP com dados históricos.
  - Salva o modelo e o escalador no S3.
  - Realiza previsões de probabilidade.

---

### **7. `xgboost_regression.py`**
- **Descrição**: Implementa um modelo de regressão utilizando o algoritmo XGBoost.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Treina o modelo XGBoost com dados históricos.
  - Salva o modelo e o escalador no S3.
  - Realiza previsões contínuas e converte para probabilidades.

---

### **8. `lstm_classification.py`**
- **Descrição**: Implementa um modelo de classificação utilizando Redes Neurais Recorrentes (LSTM).
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Treina o modelo LSTM com dados históricos.
  - Salva o modelo e o escalador no S3.
  - Realiza previsões de probabilidade.

---

### **9. `stacking_ensemble.py`**
- **Descrição**: Implementa um modelo de ensemble que combina as previsões dos modelos `LSTM`, `XGBoost` e `MLP` utilizando uma Rede Neural como meta-modelo.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Gera meta-features a partir das previsões dos modelos base.
  - Treina o meta-modelo com as meta-features.
  - Salva o meta-modelo e o escalador no S3.

---

### **10. `rl_ppo_model.py`**
- **Descrição**: Implementa um modelo de Aprendizado por Reforço utilizando o algoritmo PPO (Proximal Policy Optimization).
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Define um ambiente personalizado para negociação de criptomoedas.
  - Treina o modelo PPO com dados históricos.
  - Realiza previsões e coleta métricas de desempenho.

---

### **11. `rdsConnect.py`**
- **Descrição**: Gerencia a conexão com o banco de dados PostgreSQL utilizando o AWS RDS.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Cria um pool de conexões para o banco de dados.
  - Fornece métodos para obter e liberar conexões.

---

### **12. `rds_dml_functions.py`**
- **Descrição**: Contém funções para executar operações DML (Data Manipulation Language) no banco de dados.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Insere métricas de desempenho no banco de dados.
  - Limpa tabelas específicas.

---

### **13. `binance_client.py`**
- **Descrição**: Configura o cliente da Binance para interagir com a API da exchange.
- **Status**: **Não utilizado na estratégia atual**.
- **Funcionalidade**:
  - Inicializa o cliente da Binance utilizando chaves de API.

---

### **14. `parameterServiceAWS.py`**
- **Descrição**: Função para obter parâmetros do AWS SSM Parameter Store.
- **Status**: **Não utilizado na estratégia atual**.
- **Funcionalidade**:
  - Recupera parâmetros armazenados no SSM Parameter Store.

---

---

### **15. `create_endpoint_sagemaker.py`**
- **Descrição**: Classe para gerenciar a criação, treinamento e implantação de modelos no Amazon SageMaker.
- **Status**: **Não utilizado na estratégia atual**.
- **Funcionalidade**:
  - Configura um estimador XGBoost com hiperparâmetros específicos.
  - Treina o modelo utilizando dados armazenados no S3.
  - Implanta o modelo em um endpoint do SageMaker para realizar previsões.
  - Inclui métodos para excluir configurações de endpoints existentes antes de criar novos.

---

### **16. `read_from_s3.py`**
- **Descrição**: Função assíncrona para leitura de arquivos Parquet armazenados no S3.
- **Status**: **Em uso**.
- **Funcionalidade**:
  - Lê dados refinados de um bucket S3 com base em parâmetros como `ticker`, `timeframe` e `period`.
  - Retorna os dados como um DataFrame do Pandas.
  - Lida com erros de chave ausente no S3 e encerra o programa com uma mensagem de erro apropriada.

## Fluxo de Execução

1. **Carregamento de Dados**:
   - Os dados históricos são carregados do Data Lake ou de um arquivo Parquet local.

2. **Treinamento de Modelos**:
   - Os modelos `XGBoost`, `LSTM` e `MLP` são treinados com os dados históricos.
   - Um modelo de ensemble combina as previsões dos modelos base.

3. **Backtest**:
   - O backtest é realizado em janelas temporais de 30 dias.
   - Métricas de desempenho são calculadas e armazenadas.

4. **Aprendizado por Reforço**:
   - O modelo PPO é treinado com as previsões do ensemble e realiza negociações simuladas.

5. **Armazenamento de Resultados**:
   - Métricas de desempenho são salvas no banco de dados.
   - Modelos e parâmetros de escalonamento são salvos no S3.

---

## Observações

- **Códigos Desabilitados**:
  - `performance_models.py`: Calcula indicadores e gera relatórios, mas não está sendo utilizado na estratégia atual.
  - `binance_client.py`: Configura o cliente da Binance, mas não é necessário para o backtest.
  - `parameterServiceAWS.py`: Recupera parâmetros do SSM, mas não está sendo utilizado.

- **Dependências**:
  - AWS S3 para armazenamento de modelos e datasets.
  - AWS RDS para armazenamento de métricas.
  - Bibliotecas de ML/DL como `scikit-learn`, `xgboost`, `tensorflow` e `stable-baselines3`.

---

## Como Executar

1. Certifique-se de que todas as dependências estão instaladas.
2. Configure as credenciais da AWS para acesso ao S3 e RDS.
3. Execute o arquivo `main.py`:
   ```bash
   python main.py