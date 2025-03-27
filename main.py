import os
import sys
import asyncio
from estrategia.estrategia1 import Estrategia1

# Adicione o caminho do diretório pai ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def main():
    #estrategia = Estrategia1(bucket_name='datalake-tc3', ticker='BTCUSDT', timeframe='1h', period='1ano')
    estrategia = Estrategia1(bucket_name='datalake-tc3', bucket_name_model='models-bucket-tc3', ticker='BTCUSDT', timeframe='15m', period='1ano')
    result = await estrategia.run_backtest()

if __name__ == '__main__':
    asyncio.run(main())
    