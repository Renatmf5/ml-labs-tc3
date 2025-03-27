from services.rdsConnect import db_instance
import psycopg2

def execute_query(query, params):
    connection = db_instance.get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            connection.commit()
    except psycopg2.OperationalError as e:
        print(f"Erro operacional: {e}")
        if connection and not connection.closed:
            connection.rollback()
    except Exception as e:
        print(f"Erro ao executar a consulta: {e}")
        if connection and not connection.closed:
            connection.rollback()
    finally:
        db_instance.release_connection(connection)

def clear_trade_metrics():
    query = "DELETE FROM trade_metrics"
    execute_query(query, None)
    
def insert_trade_metrics(df,batch_id):
    clear_trade_metrics()
    
    query = """
    INSERT INTO trade_metrics (
        batch_id, total_trades, win_rate, avg_trade_duration, avg_profit_pct, avg_loss_pct, final_capital, max_gain, max_loss, max_drawdown, max_consecutive_gains, max_consecutive_losses, stop_loss_count, take_profit_count, long_trades_count, short_trades_count, start_time, end_time,win_count,loss_count, final_gross_capital)
     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    connection = db_instance.get_connection()
    try:
        with connection.cursor() as cursor:
            for index, row in df.iterrows():
                params = (
                     batch_id, row['total_trades'], row['win_rate'], row['avg_trade_duration'], row['avg_profit_pct'], row['avg_loss_pct'], row['final_capital'], row['max_gain'], row['max_loss'], row['max_drawdown'], row['max_consecutive_gains'], row['max_consecutive_losses'], row['stop_loss_count'], row['take_profit_count'], row['long_trades_count'], row['short_trades_count'], row['start_time'], row['end_time'], row['win_count'], row['loss_count'], row['final_gross_capital']
                )
                cursor.execute(query, params)
            connection.commit()
    except psycopg2.OperationalError as e:
        print(f"Erro operacional: {e}")
        if connection and not connection.closed:
            connection.rollback()
    except Exception as e:
        print(f"Erro ao executar a consulta: {e}")
        if connection and not connection.closed:
            connection.rollback()
    finally:
        db_instance.release_connection(connection) 