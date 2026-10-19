"""GridSearchCV, evaluación independiente y trazabilidad MLflow (sin GPU)."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
import json, time, hashlib, platform, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import sklearn, mlflow, mlflow.sklearn
from mlflow.models import infer_signature
from datos_prep import ROOT, sha, prepare

SEED=42

def scores(y,p,proba,labels):
    return {'accuracy':accuracy_score(y,p),'precision_macro':precision_score(y,p,average='macro',zero_division=0),
      'recall_macro':recall_score(y,p,average='macro',zero_division=0),'f1_macro':f1_score(y,p,average='macro',zero_division=0),
      'roc_auc_ovr_macro':roc_auc_score(y,proba,labels=labels,multi_class='ovr',average='macro')}

def main():
    quality=prepare();out=ROOT/'resultados';out.mkdir(exist_ok=True)
    train=pd.read_csv(ROOT/'datos/datos_limp/train.csv');test=pd.read_csv(ROOT/'datos/datos_limp/test.csv')
    X=train['text'];y=train['category'];Xt=test['text'];yt=test['category']; labels=sorted(y.unique())
    assert set(labels)==set(yt.unique())
    # El registro usa SQLite y artefactos locales; se reconstruye al ejecutar en otro equipo.
    uri='sqlite:///'+str(ROOT/'mlflow.db'); mlflow.set_tracking_uri(uri)
    name_exp='SentimentAI_Banking_v2_BANKING77'
    client=mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(name_exp) is None:
        client.create_experiment(name_exp,artifact_location=(ROOT/'mlruns').as_uri())
    exp=mlflow.set_experiment(name_exp)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'fuentes/datos_prep.py').read_bytes()).hexdigest()
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    gitrev=git.stdout.strip() if git.returncode==0 else 'sin_commit_local'
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    grids={
      'Regresion_logistica':(LogisticRegression(max_iter=1000,solver='lbfgs',random_state=SEED),{'model__C':[0.5,2.0,8.0]}),
      'Multinomial_NB':(MultinomialNB(),{'model__alpha':[0.1,0.5,1.0]})}
    metrics=[];preds={};best={};run_ids={}
    for name,(model,params) in grids.items():
        print('ENTRENANDO',name,flush=True)
        pipeline=Pipeline([('tfidf',TfidfVectorizer(min_df=2,sublinear_tf=True,max_features=30000)),('model',model)])
        params={**params,'tfidf__ngram_range':[(1,1),(1,2)]}
        search=GridSearchCV(pipeline,params,cv=cv,scoring={'f1_macro':'f1_macro','accuracy':'accuracy'},refit='f1_macro',n_jobs=1,error_score='raise',return_train_score=True)
        with mlflow.start_run(run_name=name) as parent:
            run_ids[name]=parent.info.run_id
            mlflow.set_tags({'dataset':'BANKING77','idioma':'en','tarea':'intencion_77_clases','data_version':'2.0.0','code_sha256':source_hash,'git_commit':gitrev,'seleccion':'CV_f1_macro_solo_train'})
            mlflow.log_params({'seed':SEED,'cv_folds':5,'train_sha256':quality['archivos_limpios_sha256']['train'],'test_sha256':quality['archivos_limpios_sha256']['test'],'n_train':len(train),'n_test':len(test),'max_features':30000,'min_df':2,'sublinear_tf':True})
            mlflow.log_input(mlflow.data.from_pandas(train[['text','category']],source='BANKING77 train',targets='category',name='banking77_train_v2'),context='training')
            start=time.perf_counter();search.fit(X,y);elapsed=time.perf_counter()-start
            result=pd.DataFrame(search.cv_results_);result.to_csv(out/f'cv_{name}.csv',index=False)
            for i,row in result.iterrows():
                with mlflow.start_run(run_name=f'config_{i:02d}',nested=True):
                    mlflow.log_params({k:str(v) for k,v in row['params'].items()})
                    mlflow.log_metrics({'cv_f1_macro_mean':row['mean_test_f1_macro'],'cv_f1_macro_std':row['std_test_f1_macro'],'cv_accuracy_mean':row['mean_test_accuracy'],'cv_train_f1_mean':row['mean_train_f1_macro'],'fit_seconds_mean':row['mean_fit_time']})
                    for fold in range(5):mlflow.log_metric('fold_f1_macro',row[f'split{fold}_test_f1_macro'],step=fold)
            fitted=search.best_estimator_;best[name]=fitted
            print('MEJOR CV',search.best_score_,search.best_params_,flush=True)
            p=fitted.predict(Xt);proba=fitted.predict_proba(Xt);preds[name]=p
            row={'modelo':name,'cv_f1_macro':search.best_score_,'cv_f1_std':float(result.iloc[search.best_index_]['std_test_f1_macro']),'busqueda_segundos':elapsed,**scores(yt,p,proba,labels)}
            metrics.append(row)
            mlflow.log_params({'best_'+k:str(v) for k,v in search.best_params_.items()});mlflow.log_metrics({k:v for k,v in row.items() if k!='modelo'})
            mlflow.log_artifact(str(out/f'cv_{name}.csv'),'validacion')
            mlflow.log_artifact(str(ROOT/'datos/datos_limp/calidad.json'),'datos')
            mlflow.log_artifact(str(ROOT/'datos/manifest_origen.json'),'datos')
            mlflow.log_artifacts(str(ROOT/'fuentes'),'codigo')
            model_path=out/f'modelo_{name}'
            if model_path.exists():
                import shutil;shutil.rmtree(model_path)
            mlflow.sklearn.save_model(fitted,str(model_path),signature=infer_signature(Xt.iloc[:3].to_numpy(),p[:3]))
            mlflow.log_artifacts(str(model_path),'modelo')
            # Verificación de carga del modelo registrado en el run.
            loaded=mlflow.sklearn.load_model(f'runs:/{parent.info.run_id}/modelo')
            assert np.array_equal(loaded.predict(Xt.iloc[:10]),p[:10])
            report=classification_report(yt,p,output_dict=True,zero_division=0)
            (out/f'clases_{name}.json').write_text(json.dumps(report,indent=2))
            cm=confusion_matrix(yt,p,labels=labels);pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/f'confusion_{name}.csv')
            frame=test[['id','text','category']].copy();frame['prediccion']=p;frame['score_maximo']=proba.max(axis=1);frame['correcto']=frame['category'].eq(frame['prediccion']);frame.to_csv(out/f'predicciones_{name}.csv',index=False)
            np.savez_compressed(out/f'probabilidades_{name}.npz',proba=proba,labels=labels)
            mlflow.log_artifact(str(out/f'predicciones_{name}.csv'),'evaluacion');mlflow.log_artifact(str(out/f'clases_{name}.json'),'evaluacion')
    m=pd.DataFrame(metrics);m.to_csv(out/'metricas.csv',index=False)
    # La selección se fija por CV; las métricas de test no intervienen en el ajuste.
    winner=m.sort_values(['cv_f1_macro','modelo'],ascending=[False,True]).iloc[0]['modelo']
    names=list(preds);a=np.array(preds[names[0]])==yt.to_numpy();b=np.array(preds[names[1]])==yt.to_numpy()
    only_a=int(np.sum(a & ~b));only_b=int(np.sum(~a & b));discord=only_a+only_b
    rng=np.random.default_rng(SEED);groups=[np.where(yt.to_numpy()==c)[0] for c in labels];delta=[]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);delta.append(float(a[ix].mean()-b[ix].mean()))
    stat={'modelo_A':names[0],'modelo_B':names[1],'seleccionado_por_CV':winner,'solo_A_acierta':only_a,'solo_B_acierta':only_b,'mcnemar_exacto_p':float(binomtest(only_a,discord,0.5).pvalue) if discord else 1.0,'diferencia_accuracy_A_menos_B':float(a.mean()-b.mean()),'IC95_bootstrap_pareado_estratificado_accuracy':np.quantile(delta,[.025,.975]).tolist(),'bootstrap_replicas':1000,'limite':'Comparación de aciertos en un test fijo; no demuestra rendimiento en español ni en producción. La dispersión CV no es un intervalo de confianza independiente.'}
    (out/'comparacion_estadistica.json').write_text(json.dumps(stat,indent=2))
    base=DummyClassifier(strategy='most_frequent').fit(X,y);bp=base.predict(Xt)
    (out/'baseline.json').write_text(json.dumps({'accuracy':accuracy_score(yt,bp),'f1_macro':f1_score(yt,bp,average='macro',zero_division=0),'regla':'clase_mayoritaria_train'},indent=2))
    # Visualizaciones con datos del experimento real.
    plt.rcParams.update({'font.size':10})
    fig,ax=plt.subplots(figsize=(8,4));m.set_index('modelo')[['accuracy','f1_macro','roc_auc_ovr_macro']].plot.bar(ax=ax,rot=0,ylim=(0,1));ax.set_title('BANKING77 · test independiente');fig.tight_layout();fig.savefig(out/'comparacion.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));train.groupby('category').size().sort_values().plot.bar(ax=ax);ax.set_xticks([]);ax.set_ylabel('Consultas');ax.set_title('Distribución de las 77 clases · entrenamiento limpio');fig.tight_layout();fig.savefig(out/'distribucion_clases.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(train.text.str.split().str.len(),bins=35,color='#117d91');ax.set_xlabel('Palabras por consulta');ax.set_title('Longitud del texto · entrenamiento limpio');fig.tight_layout();fig.savefig(out/'longitud_textos.png',dpi=160);plt.close(fig)
    for name in names:
        cm=pd.read_csv(out/f'confusion_{name}.csv',index_col=0).to_numpy();np.fill_diagonal(cm,0)
        pairs=sorted([(int(cm[i,j]),labels[i]+' → '+labels[j]) for i in range(len(labels)) for j in range(len(labels)) if cm[i,j]>0],reverse=True)[:10]
        fig,ax=plt.subplots(figsize=(11,5));ax.barh([v[1] for v in pairs][::-1],[v[0] for v in pairs][::-1],color='#117d91');ax.set_xlabel('Confusiones en test');ax.set_title(name+' · diez confusiones más frecuentes');fig.tight_layout();fig.savefig(out/f'errores_{name}.png',dpi=160);plt.close(fig)
    env={'python':platform.python_version(),'sklearn':sklearn.__version__,'mlflow':mlflow.__version__,'pandas':pd.__version__,'seed':SEED,'code_sha256':source_hash,'git_commit':gitrev,'tracking_uri_original':uri,'experimento_id':exp.experiment_id,'run_ids':run_ids}
    (out/'entorno.json').write_text(json.dumps(env,indent=2))
    with mlflow.start_run(run_name='Comparacion_final'):
        mlflow.set_tags({'seleccionado_por_CV':winner,'code_sha256':source_hash})
        mlflow.log_artifact(str(out/'metricas.csv'));mlflow.log_artifact(str(out/'comparacion_estadistica.json'))
        mlflow.log_artifacts(str(out),'resultados')
    print(m.to_string(index=False),flush=True);print(json.dumps(stat,indent=2),flush=True)
if __name__=='__main__':main()
