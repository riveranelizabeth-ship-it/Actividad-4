"""Preparación determinista de BANKING77. Ejecutar desde cualquier directorio."""
from pathlib import Path
from urllib.request import urlopen
import hashlib, json, re, unicodedata
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text))).strip().lower()

def prepare():
    manifest=json.loads((ROOT/'datos/manifest_origen.json').read_text())
    raw=ROOT/'datos/datos_ini'; clean=ROOT/'datos/datos_limp'; clean.mkdir(parents=True,exist_ok=True)
    frames=[]
    for split in ['train','test']:
        path=raw/f'{split}.csv'; meta=manifest['archivos'][path.name]
        if not path.exists():
            raw.mkdir(parents=True,exist_ok=True)
            path.write_bytes(urlopen(meta['url'],timeout=60).read())
        if sha(path)!=meta['sha256']: raise ValueError('El archivo original no coincide con su SHA256')
        df=pd.read_csv(path,keep_default_na=False)
        if set(df.columns)!={'text','category'}: raise ValueError('Esquema inesperado')
        df['id']=[f'{split}_{i:05d}' for i in range(len(df))]; df['split']=split
        frames.append(df)
    d=pd.concat(frames,ignore_index=True); d['text_original']=d['text'];d['text']=d['text'].map(normalize)
    d['category']=d['category'].str.strip()
    audit={'originales':len(d),'por_split_original':d['split'].value_counts().to_dict()}
    invalid=d['text'].eq('')|d['category'].eq('');audit['vacios']=int(invalid.sum());d=d.loc[~invalid].copy()
    conflicts=d.groupby('text')['category'].nunique(); bad=set(conflicts[conflicts>1].index)
    audit['textos_con_etiquetas_conflictivas']=len(bad);audit['filas_conflictivas']=int(d['text'].isin(bad).sum())
    excluded=d.loc[d['text'].isin(bad)].copy();excluded['motivo']='etiqueta_conflictiva'
    d=d.loc[~d['text'].isin(bad)].copy()
    # Mantener el test oficial: quitar del entrenamiento textos iguales a los de test.
    overlap=d['split'].eq('train') & d['text'].isin(d.loc[d['split'].eq('test'),'text'])
    audit['solapamientos_retirados_train']=int(overlap.sum())
    extra=d.loc[overlap].copy();extra['motivo']='solapamiento_con_test';excluded=pd.concat([excluded,extra]);d=d.loc[~overlap]
    dup=d.duplicated(['split','text'],keep='first'); audit['duplicados_retirados']=int(dup.sum())
    extra=d.loc[dup].copy();extra['motivo']='duplicado_normalizado';excluded=pd.concat([excluded,extra]);d=d.loc[~dup].copy()
    for split in ['train','test']:
        part=d.loc[d['split'].eq(split),['id','text','category','text_original']]
        part.to_csv(clean/f'{split}.csv',index=False)
    assert not set(d.loc[d['split'].eq('train'),'text']) & set(d.loc[d['split'].eq('test'),'text'])
    audit['limpios']=len(d);audit['por_split_limpio']=d['split'].value_counts().to_dict();audit['categorias']=int(d['category'].nunique())
    audit['archivos_limpios_sha256']={s:sha(clean/f'{s}.csv') for s in ['train','test']}
    excluded.to_csv(clean/'exclusiones.csv',index=False)
    (clean/'calidad.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit,indent=2)); return audit
if __name__=='__main__': prepare()
