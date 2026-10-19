from pathlib import Path
import sqlite3,zipfile
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'mlflow.db').exists():
    raise SystemExit('Ya existe mlflow.db. Conserva una copia o usa otro directorio antes de restaurar.')
with zipfile.ZipFile(ROOT/'evidencias/seguimiento_mlflow.zip') as z:z.extractall(ROOT)
with sqlite3.connect(ROOT/'mlflow.db') as conn:
    for table,col in [('experiments','artifact_location'),('runs','artifact_uri')]:
        for key,value in conn.execute(f'SELECT rowid,{col} FROM {table}').fetchall():
            if value and '/mlruns' in value:
                tail=value.split('/mlruns',1)[1]
                conn.execute(f'UPDATE {table} SET {col}=? WHERE rowid=?',((ROOT/'mlruns').as_uri()+tail,key))
print('Seguimiento restaurado. Desde esta carpeta: python -m mlflow ui --backend-store-uri sqlite:///mlflow.db')
