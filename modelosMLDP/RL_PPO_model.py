import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import os
import boto3

class CryptoTradingEnv(gym.Env):
    """Ambiente personalizado para treinamento de RL no mercado de criptomoedas."""
    
    def __init__(self, df, window_size, stop_loss_pct=0.02, take_profit_pct=0.02, tax=0.0008):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.tax =tax
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.initial_capital = 1000
        self.current_capital = self.initial_capital
        self.gross_capital = self.initial_capital

        # Definir espaços de ação e observação
        self.action_space = gym.spaces.Discrete(3)  # 0: Sell, 1: Buy, 2: Hold
        self.observation_space = gym.spaces.Box(
            low=-0, high=1, shape=(window_size, 22), dtype=np.float32
            #low=0, high=1, shape=(window_size, 4), dtype=np.float3
        )
        
        # Estatísticas para avaliação
        self.trades = []  # Lista de trades (entrada, saída, tipo, lucro)
        self.current_position = None  # None, "buy", "sell"
        self.entry_price = None  # Preço de entrada na posição
        self.entry_candle = None
        self.hold_count = 0  # Quantidade de vezes que a ação foi Hold
        self.stop_loss_count = 0  # Contador de trades que acionaram stop loss
        self.take_profit_count = 0  # Contador de trades que acionaram take profit
        self.long_trades_count = 0  # Contador de trades de posição long (compras)
        self.short_trades_count = 0  # Contador de trades de posição short (vendas)
        self.win_count = 0  # Contador de trades vencedores
        self.loss_count = 0  # Contador de trades perdedores


    def reset(self, seed=None, options=None):
        """Reinicia o ambiente e retorna a primeira observação."""
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.done = False
        self.total_reward = 0
        self.current_capital = self.initial_capital
        self.gross_capital = self.initial_capital
        self.trades = []
        self.current_position = None
        self.entry_price = None
        self.entry_candle = None
        self.hold_count = 0
        self.stop_loss_count = 0
        self.take_profit_count = 0
        self.long_trades_count = 0
        self.short_trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        return self._next_observation(), {}

    def _next_observation(self):
        """Retorna a observação atual baseada na janela de dados."""  
        #return self.df.iloc[self.current_step - self.window_size:self.current_step].values.astype(np.float32)
        obs = self.df[['sma_5_diff', 'sma_20_diff', 'sma_50_diff', 'ema_20_diff','mean_proportion_BTC', 'std_proportion_BTC', 'proportion_taker_BTC', 
        'z_score_BTC','cci_20', 'stc', 'roc_10', 'cmo', 'obv', 'mfi', 'tsi_stoch', 'dmi_plus',
       'dmi_minus', 'adx', 'pred_lstm_proba', 'pred_xgb_proba',
       'pred_mlp_proba', 'ensemble_signal']].iloc[self.current_step - self.window_size:self.current_step]. values.astype(np.float32)
        return obs

    def step(self, action):
        """Executa uma ação e avança para o próximo estado."""
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.done = True
            return self._next_observation(), 0, self.done, False, {}

        price = self.df['close'].iloc[self.current_step]
        low = self.df['low'].iloc[self.current_step]
        high = self.df['high'].iloc[self.current_step]
        reward = 0
        
        trade_penalty = 0.5
        
        # Verificar stop loss e take profit
        if self.current_position == "buy":
            if low <= self.entry_price * (1 - self.stop_loss_pct):
                quantity = self.current_capital / self.entry_price
                profit = (self.entry_price * (1 - self.stop_loss_pct) - self.entry_price) * quantity
                gross_profit = profit  # Sem aplicar taxa
                trade_return = (profit / self.current_capital) * 100
                tax_amount = self.entry_price * self.tax * quantity
                profit -= tax_amount
                self.trades.append({
                    "entry_index": self.entry_candle,
                    "exit_index": self.current_step,
                    "entry_price": self.entry_price,
                    "exit_price": low,
                    "return": trade_return,
                    "signal": "buy",
                    "success": False,
                    "investment_value": self.current_capital + profit
                })
                self.current_position = None
                self.entry_price = None
                self.current_capital += profit
                self.gross_capital += gross_profit  # Atualizar gross_capital
                self.stop_loss_count += 1  # Incrementar contador de stop loss
                penalty = min(abs(trade_return) * 0.5, 10)  # Penalização limitada a 10%
                reward = -10 + (trade_return * 5)
            elif high >= self.entry_price * (1 + self.take_profit_pct):
                quantity = self.current_capital / self.entry_price
                profit = (self.entry_price * (1 + self.take_profit_pct) - self.entry_price) * quantity
                gross_profit = profit  # Sem aplicar taxa
                trade_return = (profit / self.current_capital) * 100
                tax_amount = self.entry_price * self.tax * quantity
                profit -= tax_amount
                self.trades.append({
                    "entry_index": self.entry_candle,
                    "exit_index": self.current_step,
                    "entry_price": self.entry_price,
                    "exit_price": high,
                    "return": trade_return,
                    "signal": "buy",
                    "success": True,
                    "investment_value": self.current_capital + profit
                })
                self.current_position = None
                self.entry_price = None
                self.current_capital += profit
                self.gross_capital += gross_profit  # Atualizar gross_capital
                self.take_profit_count += 1  # Incrementar contador de take profit
                reward = 10 + (trade_return * 5)
        elif self.current_position == "sell":
            if high >= self.entry_price * (1 + self.stop_loss_pct):
                quantity = self.current_capital / self.entry_price
                profit = (self.entry_price - self.entry_price * (1 + self.stop_loss_pct)) * quantity
                gross_profit = profit  # Sem aplicar taxa
                trade_return = (profit / self.current_capital) * 100
                tax_amount = self.entry_price * self.tax * quantity
                profit -= tax_amount
                self.trades.append({
                    "entry_index": self.entry_candle,
                    "exit_index": self.current_step,
                    "entry_price": self.entry_price,
                    "exit_price": high,
                    "return": trade_return,
                    "signal": "sell",
                    "success": False,
                    "investment_value": self.current_capital + profit
                })
                self.current_position = None
                self.entry_price = None
                self.current_capital += profit
                self.gross_capital += gross_profit  # Atualizar gross_capital
                self.stop_loss_count += 1  # Incrementar contador de stop loss
                penalty = min(abs(trade_return) * 0.5, 10)  # Penalização limitada a 10%
                reward = -10 + (trade_return * 5)
            elif low <= self.entry_price * (1 - self.take_profit_pct):
                quantity = self.current_capital / self.entry_price
                profit = (self.entry_price - self.entry_price * (1 - self.take_profit_pct)) * quantity
                gross_profit = profit  # Sem aplicar taxa
                trade_return = (profit / self.current_capital) * 100
                tax_amount = self.entry_price * self.tax * quantity
                profit -= tax_amount
                self.trades.append({
                    "entry_index": self.entry_candle,
                    "exit_index": self.current_step,
                    "entry_price": self.entry_price,
                    "exit_price": low,
                    "return": trade_return,
                    "signal": "sell",
                    "success": True,
                    "investment_value": self.current_capital + profit
                })
                self.current_position = None
                self.entry_price = None
                self.current_capital += profit
                self.gross_capital += gross_profit  # Atualizar gross_capital
                self.take_profit_count += 1  # Incrementar contador de take profit
                reward = 10 + (trade_return * 5)
        
        
        if action == 1:  # Buy
            if self.current_position is None:
                self.current_position = "buy"
                self.entry_price = price
                self.entry_candle = self.current_step
                self.long_trades_count += 1
                reward -= trade_penalty
            elif self.current_position == "sell":  # Fecha uma venda
                quantity = self.current_capital / self.entry_price
                profit = (self.entry_price - price) * quantity
                gross_profit = profit  # Sem aplicar taxa
                trade_return = (profit / self.current_capital) * 100
                success = profit > 0
                tax_amount = self.entry_price * self.tax * quantity
                profit -= tax_amount
                trade_duration = self.current_step - self.entry_candle
                if trade_duration >= 1:
                    if trade_return > 0:  # Trade lucrativo
                        reward = trade_return * 3  # Multiplica o retorno positivo por 4
                        if trade_return > 0.5:  # Incentiva ganhos maiores que 50%
                            reward = 5 + (trade_return * 5)  # Recompensa adicional
                    elif trade_return < -0.5:  # Trade com prejuízo
                        reward = trade_return * 4
                    else: 
                        reward = trade_return * 2
                    if trade_duration > 90:
                        penalty = (trade_duration - 90) * 2  # Penalização de 0.01 por candle após 90
                        reward -= penalty
                    self.current_capital += profit
                    self.gross_capital += gross_profit  # Atualizar gross_capital
                else:
                    reward = 0  # Recompensa é o trade_return para prejuízos
                    self.current_capital += profit
                    self.gross_capital += gross_profit  # Atualizar gross_capital
                self.trades.append({
                    "entry_index": self.entry_candle,
                    "exit_index": self.current_step,
                    "entry_price": self.entry_price,
                    "exit_price": price,
                    "return": trade_return,
                    "signal": "sell",
                    "success": success,
                    "investment_value": self.current_capital + profit
                })
                self.current_position = None
                self.entry_price = None
                if success:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                    
        elif action == 0:  # Sell
            if self.current_position is None:
                self.current_position = "sell"
                self.entry_price = price
                self.entry_candle = self.current_step
                self.short_trades_count += 1
                reward -= trade_penalty
            elif self.current_position == "buy":  # Fecha uma compra
                quantity = self.current_capital / self.entry_price
                profit = (price - self.entry_price) * quantity
                gross_profit = profit  # Sem aplicar taxa
                trade_return = (profit / self.current_capital) * 100
                success = profit > 0
                tax_amount = self.entry_price * self.tax * quantity
                profit -= tax_amount
                trade_duration = self.current_step - self.entry_candle
                if trade_duration >= 1:
                    if trade_return > 0:  # Trade lucrativo
                        reward = trade_return * 3  # Multiplica o retorno positivo por 4
                        if trade_return > 0.5:  # Incentiva ganhos maiores que 50%
                            reward = 5 + (trade_return * 5)  # Recompensa adicional
                    elif trade_return < -0.5:  # Trade com prejuízo
                        reward = trade_return * 4
                    else: 
                        reward = trade_return * 2
                    if trade_duration > 90:
                        penalty = (trade_duration - 90) * 2  # Penalização de 0.01 por candle após 90
                        reward -= penalty
                    self.current_capital += profit
                    self.gross_capital += gross_profit  # Atualizar gross_capital
                else:
                    reward = 0  # Recompensa é o trade_return para prejuízos
                    self.current_capital += profit
                    self.gross_capital += gross_profit  # Atualizar gross_capital
                self.trades.append({
                    "entry_index": self.entry_candle,
                    "exit_index": self.current_step,
                    "entry_price": self.entry_price,
                    "exit_price": price,
                    "return": trade_return,
                    "signal": "buy",
                    "success": success,
                    "investment_value": self.current_capital + profit
                })
                self.current_position = None
                self.entry_price = None
                if success:
                    self.win_count += 1
                else:
                    self.loss_count += 1
        
        elif action == 2:  # Hold
            self.hold_count += 1  # Contar quantas vezes foi Hold
            

        self.total_reward += reward
        obs = self._next_observation()
        terminated = self.done
        truncated = False  # Você pode ajustar isso conforme necessário
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def get_trade_metrics(self):
        """Retorna estatísticas dos trades."""
        total_trades = len(self.trades)
        acertos = sum(1 for trade in self.trades if trade["success"])
        erros = total_trades - acertos
        avg_trade_duration = np.mean([trade["exit_index"] - trade["entry_index"] for trade in self.trades]) if total_trades > 0 else 0
        
        # Calcular média de profit e média de loss em termos de porcentagem de retorno
        avg_profit_pct = np.mean([trade["return"] for trade in self.trades if trade["return"] > 0]) if acertos > 0 else 0
        avg_loss_pct = np.mean([trade["return"] for trade in self.trades if trade["return"] < 0]) if erros > 0 else 0
        
        # Criar DataFrame de trades
        trades_df = pd.DataFrame(self.trades)

        if trades_df.empty:
            return {
                "total_trades": 0,
                "win_rate": "0.0%",
                "avg_trade_duration": "0.0",
                "avg_profit_pct": "0.000%",
                "avg_loss_pct": "0.000%",
                "final_capital": f"{self.current_capital:.2f}",
                "max_gain": "0.0%",
                "max_loss": "0.0%",
                "max_drawdown": "0.0%",
                "max_consecutive_gains": 0,
                "max_consecutive_losses": 0,
                "stop_loss_count": self.stop_loss_count,
                "take_profit_count": self.take_profit_count,
                "long_trades_count": self.long_trades_count,
                "short_trades_count": self.short_trades_count,
                "win_count": self.win_count,
                "loss_count": self.loss_count,
                "final_gross_capital": f"{self.gross_capital:.2f}"
                #"buy_win_rate": "0.0%",
                #"sell_win_rate": "0.0%"
            }

        # Maior ganho
        max_gain = trades_df['return'].max()

        # Maior perda
        max_loss = trades_df['return'].min()

        # Max drawdown
        cumulative_return = trades_df['return'].cumsum()
        running_max = cumulative_return.cummax()
        drawdown = running_max - cumulative_return
        max_drawdown = drawdown.max()

        # Maior número de trades com ganhos consecutivos
        trades_df['gain'] = trades_df['return'] > 0
        max_consecutive_gains = trades_df['gain'].astype(int).groupby(trades_df['gain'].ne(trades_df['gain'].shift()).cumsum()).cumsum().max()

        # Maior número de trades com perdas consecutivas
        trades_df['loss'] = trades_df['return'] < 0
        max_consecutive_losses = trades_df['loss'].astype(int).groupby(trades_df['loss'].ne(trades_df['loss'].shift()).cumsum()).cumsum().max()

        # Porcentagem de trades lucrativos para BUY
        buy_trades = trades_df[trades_df['signal'] == 'buy']
        buy_win_rate = (buy_trades['success'].mean() * 100) if not buy_trades.empty else 0

        # Porcentagem de trades lucrativos para SELL
        sell_trades = trades_df[trades_df['signal'] == 'sell']
        sell_win_rate = (sell_trades['success'].mean() * 100) if not sell_trades.empty else 0

        return {
            "total_trades": total_trades,
            "win_rate": f"{(acertos / total_trades * 100):.1f}" if total_trades > 0 else "0.0",
            "avg_trade_duration": f"{avg_trade_duration:.1f}",
            "avg_profit_pct": f"{avg_profit_pct:.3f}" if acertos > 0 else "0.000",
            "avg_loss_pct": f"{avg_loss_pct:.3f}" if erros > 0 else "0.000",
            "final_capital": f"{self.current_capital:.2f}",
            "max_gain": f"{max_gain:.1f}",
            "max_loss": f"{max_loss:.1f}",
            "max_drawdown": f"{max_drawdown:.1f}",
            "max_consecutive_gains": max_consecutive_gains,
            "max_consecutive_losses": max_consecutive_losses,
            "stop_loss_count": self.stop_loss_count,
            "take_profit_count": self.take_profit_count,
            "long_trades_count": self.long_trades_count,
            "short_trades_count": self.short_trades_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "final_gross_capital": f"{self.gross_capital:.2f}"
            #"buy_win_rate": f"{buy_win_rate:.1f}%",
            #"sell_win_rate": f"{sell_win_rate:.1f}%"
        }

class CryptoTradingModel:
    """Modelo de aprendizado por reforço usando PPO para negociação de criptomoedas."""
    
    def __init__(self, ticker, bucket):
        self.ticker = ticker
        self.bucket = bucket
        self.subpasta_modelo = f'models/{ticker}/rl_ppo'
        self.model_path = os.path.join("modelosMLDP", "dataModels", "ppo_trading_model.zip")

    def train_ppo(self, train_data):
        """Treina o modelo PPO com os dados fornecidos."""
        window_size = 15
        env = CryptoTradingEnv(train_data, window_size)  # Criar ambiente diretamente
        
        check_env(env)  # Verifica se o ambiente está correto ✅
        
        env = DummyVecEnv([lambda: env])  # Agora podemos criar o DummyVecEnv
        
        model = PPO('MlpPolicy', env, 
            verbose=1, 
            learning_rate= 0.0003,  # Decaimento linear
            n_steps=4096,                       # Reduzido para maior estabilidade
            gamma=0.996,                         # Ajustado para priorizar recompensas futuras
            gae_lambda=0.95,                    # Mantido no padrão
            clip_range=0.1,                     # Aumentado para maior exploração
            ent_coef=0.01)                      # Incentivar exploração
        model.learn(total_timesteps=500000)
        model.save(self.model_path)
        
        print(env.envs[0])
        # Obter e imprimir métricas de trade após o treinamento
        metrics = env.envs[0].get_trade_metrics()
        metrics_df = pd.DataFrame([metrics])
        print("Métricas de trade após o treinamento:")
        print(metrics_df)
        
        # Salvar o modelo no S3
        s3_client = boto3.client('s3')
        #s3_client.upload_file(self.model_path, self.bucket, f'{self.subpasta_modelo}/ppo_trading_model.zip')

        return model

    def load_model_ppo(self):
        """Carrega um modelo PPO treinado."""
        model = PPO.load(self.model_path)
        return model

    def predict_ppo(self, test_data):
        """Realiza previsões usando um modelo treinado e coleta métricas."""
        window_size = 15
        env = CryptoTradingEnv(test_data, window_size)  # Criamos diretamente o ambiente
        env = DummyVecEnv([lambda: env])  # Adicionamos o wrapper

        model = self.load_model_ppo()

        obs = env.reset()
        actions = []
        metrics = []
        
        print("Iniciando previsão...")

        for step in range(len(test_data) - window_size):
            action, _states = model.predict(obs)
            obs, rewards, done, truncated = env.step(action)
            actions.append(action)
            metrics.append(env.envs[0].get_trade_metrics())
            

            if done:
                print(f"Predição interrompida no passo {step}")
                metrics.pop()
                metrics = metrics[-1]
                metrics_df = pd.DataFrame([metrics])
                break  # Se o ambiente indicar que acabou, saímos do loop
            

        return metrics_df # Retorna estatísticas dos trades