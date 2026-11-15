import csv
import io
import json
import re
import statistics

def tokens(text):
    return re.findall(r"[^\W\d_]+", text.casefold())

def evaluar_drift(raw):
    rows = list(csv.DictReader(io.StringIO(raw.lstrip("\ufeff"))))
    reference = [tokens(row["comentario"]) for row in rows[:150]]
    calibration = [tokens(row["comentario"]) for row in rows[150:200]]
    control = [tokens(row["comentario"]) for row in rows[200:]]
    drift = [words + ["tokenizacion", "postcuantica", "interoperabilidad",
                      "criptografica", "custodia", "descentralizada"] for words in control]
    vocab = {word for words in reference for word in words}

    def indicators(batch):
        total = sum(map(len, batch))
        return {
            "comentarios": len(batch),
            "mediana_palabras": statistics.median(map(len, batch)),
            "fraccion_fuera_vocabulario": sum(word not in vocab for words in batch
                                             for word in words) / total,
        }

    base = indicators(calibration)
    result = {"vocabulario_comentarios": len(reference), "calibracion": base}
    for name, batch in (("control", control), ("cambio_sintetico", drift)):
        current = indicators(batch)
        length_delta = abs(current["mediana_palabras"] / base["mediana_palabras"] - 1)
        vocabulary_delta = current["fraccion_fuera_vocabulario"] - base["fraccion_fuera_vocabulario"]
        result[name] = {
            **current,
            "cambio_relativo_longitud": length_delta,
            "incremento_fuera_vocabulario": vocabulary_delta,
            "alerta_revision": length_delta >= .30 or vocabulary_delta >= .15,
        }
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    print(json.dumps(evaluar_drift(Path(sys.argv[1]).read_text(encoding="utf-8-sig")), indent=2))
