# PGE Ekstraliga 2027 — symulacja Monte Carlo

Prognoza sezonu PGE Ekstraligi 2027 na podstawie składów meczowych i ratingów Empirical Bayes (10 000 iteracji).

## Wymagania

- Python 3.10+
- zależności z `requirements.txt`

```bash
pip install -r requirements.txt
```

## Uruchomienie

W katalogu repozytorium:

```bash
python simulate_ekstraliga_2027.py
```

Skrypt czyta `sklady_meczowe_prognoza_2027.xlsx` (arkusz `Szczegóły`) i zapisuje:

| Plik | Opis |
|------|------|
| `wyniki_symulacji_2027.csv` | tabela zbiorcza (% play-off, złoto, medal, baraż, spadek) |
| `prognoza_2027.png` | heatmapa P(miejsce końcowe) |

## Pliki w repo

- `simulate_ekstraliga_2027.py` — kod symulacji
- `sklady_meczowe_prognoza_2027.xlsx` — składy meczowe / wejście
- `requirements.txt` — zależności Pythona
