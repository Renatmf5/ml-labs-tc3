from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer
import matplotlib.pyplot as plt
import pandas as pd
import boto3

class PerformanceAnalyzer:
    def __init__(self, Simulation_real_time_df, estrategia, ticker, bucket_name):
        self.trades_df = Simulation_real_time_df
        self.estrategia = estrategia
        self.ticker = ticker
        self.bucket_name = bucket_name


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
        
        
        # Adicionar investimento inicial e final
        indicators['initial_investment'] = 1000
        indicators['final_investment'] = self.trades_df['investment_value'].iloc[-1]      
    
        # Novos indicadores: Trades por dia da semana
        self.trades_df['weekday'] = self.trades_df['entry_timestamp'].dt.day_name()
        trades_by_weekday = self.trades_df.groupby(['weekday', 'success']).size().unstack(fill_value=0)
        
        indicators['trades_by_weekday'] = trades_by_weekday.to_dict()
        
        # Novo indicador: Retorno por mês
        self.trades_df['month'] = self.trades_df['entry_timestamp'].dt.to_period('M')
        monthly_investment = self.trades_df.groupby('month')['investment_value'].last()
        monthly_return = monthly_investment.pct_change().fillna(0) * 100
        indicators['monthly_return'] = monthly_return.to_dict()
        
        return indicators
    
    def plot_cumulative_return(self):
         # Ordenar trades_df por exit_timestamp
        self.trades_df = self.trades_df.sort_values(by='exit_timestamp')
        
        # Calcular o retorno acumulado em porcentagem para trades_df
        self.trades_df['cumulative_return'] = self.trades_df['return'].cumsum() * 100
        
        # Encontrar o timestamp inicial do self.trades_df
        initial_timestamp = self.trades_df['entry_timestamp'].iloc[0]
        
        # Filtrar estrategia.df para incluir apenas os dados a partir do timestamp inicial
        estrategia_filtered_df = self.estrategia.df[self.estrategia.df['timestamp'] >= initial_timestamp].copy()
        
        # Ordenar estrategia_filtered_df por timestamp
        estrategia_filtered_df = estrategia_filtered_df.sort_values(by='timestamp')
        
        # Calcular o retorno acumulado em porcentagem para estrategia_filtered_df
        estrategia_filtered_df['cumulative_return'] = estrategia_filtered_df['retorno_candle'].cumsum() * 100
        
        # Plotar os retornos acumulados
        plt.figure(figsize=(10, 6))
        plt.plot(self.trades_df['exit_timestamp'], self.trades_df['cumulative_return'], label='Retorno Acumulado dos Trades', linestyle='-')
        plt.plot(estrategia_filtered_df['timestamp'], estrategia_filtered_df['cumulative_return'], label='Retorno Acumulado da Estratégia', linestyle='--')
        plt.xlabel('Data')
        plt.ylabel('Retorno Acumulado (%)')
        plt.title('Comparação de Retorno Acumulado')
        plt.legend()
        plt.grid(True)
        
        # Salvar o gráfico em um arquivo
        plt.savefig('analytics/reports/image/retorno_acumulado.png')
        plt.close()
        
        
    def plot_weekly_investment_value(self):
        # Converter entry_timestamp para datetime se necessário
        self.trades_df['entry_timestamp'] = pd.to_datetime(self.trades_df['entry_timestamp'])
        
        # Resample para obter o valor do investimento semanal
        weekly_investment = self.trades_df.resample('W', on='entry_timestamp')['investment_value'].last().ffill()
        
        # Plotar o valor do investimento semanal
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_investment.index, weekly_investment.values, label='Valor do Investimento Semanal', linestyle='-')
        plt.xlabel('Data')
        plt.ylabel('Valor do Investimento')
        plt.title('Valor do Investimento Semanal')
        plt.legend()
        plt.grid(True)
        
        # Salvar o gráfico em um arquivo
        plt.savefig('analytics/reports/image/valor_investimento_semanal.png')
        plt.close()
       
    def generate_pdf_report(self, indicators):
        doc = SimpleDocTemplate("analytics/reports/performance_report.pdf", pagesize=letter)
        elements = []

        # Adicionar data de início e fim do backtest
        start_date = self.trades_df['entry_timestamp'].min().strftime('%Y-%m-%d')
        end_date = self.trades_df['exit_timestamp'].max().strftime('%Y-%m-%d')
        elements.append(Table([[f"Data de Início: {start_date}", f"Data de Fim: {end_date}"]]))
        elements.append(Spacer(1, 20))

        # Renomear indicadores para nomes comerciais
        renamed_indicators = {
            'total_trades': 'Total de Trades',
            'successful_trades': 'Trades Bem Sucedidos',
            'unsuccessful_trades': 'Trades Mal Sucedidos',
            'avg_gain_success': 'Ganho Médio Bem Sucedido (%)',
            'avg_loss_failure': 'Perda Média Mal Sucedido (%)',
            'max_gain': 'Maior Ganho (%)',
            'max_loss': 'Maior Perda (%)',
            'max_drawdown': 'Máxima Redução (%)',
            'max_consecutive_gains': 'Máximo Consecutivos Ganhos',
            'max_consecutive_losses': 'Máximo Consecutivos Perdas',
            'success_rate': 'Taxa de Sucesso (%)',
            'initial_investment': 'Investimento Inicial',
            'final_investment': 'Investimento Final'
        }

         # Adicionar tabela de indicadores gerais
        general_data = [[renamed_indicators[key], f"{value:.2f}" if isinstance(value, (int, float)) else value] for key, value in indicators.items() if key not in ['monthly_return','trades_by_weekday']]
        general_table = Table(general_data, colWidths=[120, 95])
        general_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                           ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                           ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                           ('FONTSIZE', (0, 0), (-1, -1), 8),
                                           ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                           ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                           ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

        # Adicionar imagem do gráfico de retorno acumulado
        retorno_acumulado_img = Image('analytics/reports/image/retorno_acumulado.png', width=230, height=200)

        # Adicionar tabela de indicadores e gráfico lado a lado
        elements.append(Table([[general_table, retorno_acumulado_img]]))

        # Adicionar espaçamento
        elements.append(Spacer(1, 20))

        # Adicionar tabela de retorno por mês em formato horizontal
        monthly_return_data = [["Mês"] + [str(month) for month in indicators['monthly_return'].keys()]]
        monthly_return_data.append(["Retorno(%)"] + [f"{monthly_return:.2f}%" for monthly_return in indicators['monthly_return'].values()])
        monthly_return_table = Table(monthly_return_data, colWidths=[40] * len(monthly_return_data[0]))
        monthly_return_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                                  ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                                  ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                                  ('FONTSIZE', (0, 0), (-1, -1), 8),
                                                  ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                                  ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                                  ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        elements.append(monthly_return_table)

        # Adicionar espaçamento
        elements.append(Spacer(1, 10))

        # Adicionar imagem do valor do investimento semanal
        elements.append(Image('analytics/reports/image/valor_investimento_semanal.png', width=500, height=250))

        # Construir o PDF
        doc.build(elements)
        
    def upload_report_to_s3(self):
        s3_client = boto3.client('s3')
        file_path = "analytics/reports/performance_report.pdf"
        s3_key = f"models/{self.ticker}/backtestReport/performance_report.pdf"
        s3_client.upload_file(file_path, self.bucket_name, s3_key)
        
        

    def analyze_performance(self):
        indicators = self.calculate_indicators()
        self.plot_cumulative_return()
        self.plot_weekly_investment_value()
        self.generate_pdf_report(indicators)
        self.upload_report_to_s3()
        
        """
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
            elif key == 'monthly_return':
                print("Retorno por mês:")
                print(f"{'Month':<10} {'Return (%)':<10}")
                for month, monthly_return in value.items():
                    month_str = str(month)  # Converter o objeto Period para string
                    print(f"{month_str:<10} {monthly_return:<10.2f}")
            elif key in ['sharpe_ratio', 'total_trades', 'successful_trades', 'unsuccessful_trades', 'max_consecutive_gains', 'max_consecutive_losses']:
                print(f"{key}: {value:.2f}")
            elif key in ['initial_investment', 'final_investment']:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.2f}%")
        
        """