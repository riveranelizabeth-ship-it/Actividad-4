"""Reproduce solicitudes registradas y verifica alertas y recuperación."""
import csv
import json
from pathlib import Path

LIMIT_MS = 500
SLO = 0.95
RECALL_MIN = 0.85
REQUIRED = 2


def simulate(rows):
    if len(rows) != 300 or sum(r["etiqueta"] == "negativo" for r in rows) != 100:
        raise ValueError("Se requieren las 300 solicitudes de referencia, con 100 negativos.")
    windows, events = [], []
    for incident in ("latencia", "calidad"):
        active = False
        bad_streak = good_streak = 0
        profile = "estable"
        for step in range(1, 8):
            # Ventana 2: pico aislado. Ventanas 4 y 5: falla sostenida.
            if step in (2, 4):
                profile = "demora_inyectada" if incident == "latencia" else "variante_degradada"
            if step == 3:
                profile = "estable"
            durations = [
                float(r["duracion_ms"]) + (700 if profile == "demora_inyectada" else 0)
                for r in rows
            ]
            predictions = [
                "neutral" if profile == "variante_degradada"
                and r["prediccion"] == "negativo" and i % 2 == 0 else r["prediccion"]
                for i, r in enumerate(rows)
            ]
            failed = sum(d > LIMIT_MS or r["estado"] != "ok" for d, r in zip(durations, rows))
            recall = sum(
                p == "negativo" and r["etiqueta"] == "negativo"
                for p, r in zip(predictions, rows)
            ) / 100
            allowed = round(len(rows) * (1 - SLO))
            bad = failed > allowed if incident == "latencia" else recall < RECALL_MIN
            bad_streak = bad_streak + 1 if bad else 0
            good_streak = 0 if bad else good_streak + 1
            rank = (len(durations) - 1) * .95
            ordered = sorted(durations)
            index = int(rank)
            p95 = ordered[index] + (rank - index) * (ordered[index + 1] - ordered[index])
            windows.append({
                "incidente": incident, "ventana": step, "perfil": profile,
                "p95_ms": round(p95, 4), "cumplimiento_slo": (len(rows) - failed) / len(rows),
                "consumo_presupuesto": failed / allowed, "recall_negativo": recall,
            })
            if bad and not active and bad_streak >= REQUIRED:
                active = True
                action = "retirar demora de prueba" if incident == "latencia" else "rollback a perfil estable"
                events.append({"incidente": incident, "ventana": step, "evento": "alerta",
                               "prioridad": "P1", "accion": action, "antes": profile,
                               "despues": "estable"})
                profile = "estable"
            if active and good_streak >= REQUIRED:
                active = False
                events.append({"incidente": incident, "ventana": step, "evento": "cierre",
                               "accion": "recuperación verificada"})
    return windows, events


def verify(windows, events):
    for incident in ("latencia", "calidad"):
        observed = [(e["evento"], e["ventana"]) for e in events if e["incidente"] == incident]
        assert observed == [("alerta", 5), ("cierre", 7)], observed
        baseline, *_, recovered = [w for w in windows if w["incidente"] == incident]
        for key in ("p95_ms", "cumplimiento_slo", "consumo_presupuesto", "recall_negativo"):
            assert baseline[key] == recovered[key], (incident, key)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8-sig") as source:
        windows, events = simulate(list(csv.DictReader(source)))
    verify(windows, events)
    args.output_path.write_text(
        json.dumps({"ventanas": windows, "eventos": events}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
