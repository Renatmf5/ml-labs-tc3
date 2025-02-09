import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


class PerformanceAnalyzer:
    def __init__(self, Simulation_real_time_df, estrategia):
        self.trades_df = Simulation_real_time_df
        self.estrategia = estrategia


    def calculate_indicators(self):
        indicators = {}
    
        # Número total de trades
        indicators['total_trades'] = len(self.trades_df)
        
        # Número de trades com sucesso
        indicators['successful_trades'] = self.trades_df['success'].sum()
        
        # Número de trades sem sucesso
        indicators['unsuccessful_trades'] = indicators['total_trades'] - indicators['successful_trades']
        
        # Ganho médio por trade com sucesso
        indicators['avg_gain_success'] = self.trades_df[self.trades_df['success']]['return'].mean() * 100
        
        # Perda média por trade sem sucesso
        indicators['avg_loss_failure'] = self.trades_df[~self.trades_df['success']]['return'].mean() * 100
        
        # Maior ganho
        indicators['max_gain'] = self.trades_df['return'].max() * 100
        
        # Maior perda
        indicators['max_loss'] = self.trades_df['return'].min() * 100
        
        # Max drawdown
        cumulative_return = self.trades_df['return'].cumsum()
        running_max = cumulative_return.cummax()
        drawdown = running_max - cumulative_return
        indicators['max_drawdown'] = drawdown.max() * 100
        
        # Maior número de trades com ganhos consecutivos
        self.trades_df['gain'] = self.trades_df['return'] > 0
        indicators['max_consecutive_gains'] = self.trades_df['gain'].astype(int).groupby(self.trades_df['gain'].ne(self.trades_df['gain'].shift()).cumsum()).cumsum().max()
        
        # Maior número de trades com perdas consecutivas
        self.trades_df['loss'] = self.trades_df['return'] < 0
        indicators['max_consecutive_losses'] = self.trades_df['loss'].astype(int).groupby(self.trades_df['loss'].ne(self.trades_df['loss'].shift()).cumsum()).cumsum().max()
        
        # Taxa de sucesso
        indicators['success_rate'] = (indicators['successful_trades'] / indicators['total_trades']) * 100
        
        # Retorno médio por trade
        indicators['avg_return_per_trade'] = self.trades_df['return'].mean()
        
        # Retorno total da estratégia
        indicators['total_return'] = cumulative_return.iloc[-1] * 100
        
        # Sharpe Ratio
        # Supondo que os retornos são em um timeframe de 5 minutos e a taxa livre de risco anual é 1%
        risk_free_rate_annual = 0.01
        minutes_per_year = 365 * 24 * 60  # 365 dias por ano, 24 horas por dia, 60 minutos por hora
        risk_free_rate_5min = (1 + risk_free_rate_annual) ** (5 / minutes_per_year) - 1
        indicators['sharpe_ratio'] = (indicators['avg_return_per_trade'] - (risk_free_rate_5min)) / self.trades_df['return'].std()
        
        # Adicionar investimento inicial e final
        indicators['initial_investment'] = 1000
        indicators['final_investment'] = self.trades_df['investment_value'].iloc[-1]
        
        
    
        # Novos indicadores: Trades por dia da semana
        self.trades_df['weekday'] = self.trades_df['entry_timestamp'].dt.day_name()
        trades_by_weekday = self.trades_df.groupby(['weekday', 'success']).size().unstack(fill_value=0)
        
        indicators['trades_by_weekday'] = trades_by_weekday.to_dict()
        
        return indicators
    
    def plot_cumulative_return(self):
        # Plotar o retorno acumulado ao longo do tempo
        self.trades_df['cumulative_return'] = self.trades_df['return'].cumsum()
        
        # Encontrar o timestamp inicial do self.trades_df
        initial_timestamp = self.trades_df['entry_timestamp'].iloc[0]
        
        # Filtrar self.estrategia.df para incluir apenas os dados a partir do timestamp inicial
        estrategia_filtered_df = self.estrategia.df[self.estrategia.df['timestamp'] >= initial_timestamp].copy()
        
        # Calcular o retorno acumulado geral do DataFrame
        estrategia_filtered_df['cumulative_return'] = estrategia_filtered_df['retorno_candle'].cumsum()
        
         # Plotar os retornos acumulados
        plt.figure(figsize=(10, 6))
        plt.plot(self.trades_df['exit_timestamp'], self.trades_df['cumulative_return'], label='Retorno Acumulado da Estratégia', marker='o')
        plt.plot(estrategia_filtered_df['timestamp'], estrategia_filtered_df['cumulative_return'], label='Retorno Acumulado Geral', linestyle='--')
        plt.xlabel('Índice')
        plt.ylabel('Retorno Acumulado')
        plt.title('Comparação de Retorno Acumulado')
        plt.legend()
        plt.grid(True)
        
        # Salvar o gráfico em um arquivo
        plt.savefig('analytics/reports/image/retorno_acumulado.png')
        plt.close()
    
       
    def generate_pdf_report(self, indicators):
        c = canvas.Canvas("analytics/reports/performance_report.pdf", pagesize=letter)
        width, height = letter
        y = height - 30

        c.drawString(30, y, "Relatório de Performance")
        y -= 20

        for key, value in indicators.items():
            if key == 'trades_by_weekday':
                c.drawString(30, y, "Trades por dia da semana:")
                y -= 20
                for day, counts in value.items():
                    gains = counts.get(True, 0)
                    losses = counts.get(False, 0)
                    total = gains + losses
                    c.drawString(30, y, f"{day}: {gains} gains, {losses} losses, total {total} trades")
                    y -= 20
            elif key in ['sharpe_ratio', 'total_trades', 'successful_trades', 'unsuccessful_trades', 'max_consecutive_gains', 'max_consecutive_losses']:
                c.drawString(30, y, f"{key}: {value:.2f}")
                y -= 20
            elif key in ['initial_investment', 'final_investment']:
                c.drawString(30, y, f"{key}: {value:.2f}")
                y -= 20
            else:
                c.drawString(30, y, f"{key}: {value:.2f}%")
                y -= 20

        c.save()

    def analyze_performance(self):
                
        indicators = self.calculate_indicators()
        
        for key, value in indicators.items():
            if key == 'trades_by_weekday':
                print("Trades por dia da semana:")
                print(f"{'Day':<10} {'Gains':<10} {'Losses':<10} {'Total':<10}")
                total_gains = 0
                total_losses = 0
                for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                    gains = value.get(True, {}).get(day, 0)
                    losses = value.get(False, {}).get(day, 0)
                    total = gains + losses
                    gain_percentage = (gains / total * 100) if total > 0 else 0
                    total_gains += gains
                    total_losses += losses
                    print(f"{day:<10} {gains:<10} {losses:<10} {total:<10} {gain_percentage:<10.2f}")
                total_trades = total_gains + total_losses
                total_gain_percentage = (total_gains / total_trades * 100) if total_trades > 0 else 0
                print(f"{'Total':<10} {total_gains:<10} {total_losses:<10} {total_trades:<10} {total_gain_percentage:<10.2f}")
            elif key in ['sharpe_ratio', 'total_trades', 'successful_trades', 'unsuccessful_trades', 'max_consecutive_gains', 'max_consecutive_losses']:
                print(f"{key}: {value:.2f}")
            elif key in ['initial_investment', 'final_investment']:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.2f}%")
        
        self.plot_cumulative_return()
        self.generate_pdf_report(indicators)