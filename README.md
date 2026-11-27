# PGE Ekstraliga 2027 — symulacja Monte Carlo

Eksperymentalny model Monte Carlo sezonu PGE Ekstraligi 2027 (10 000 iteracji) na podstawie założonych składów meczowych i ratingów Empirical Bayes.

Wynikowe procenty to **symulacja przy przyjętych założeniach**, nie skalibrowana prognoza bukmacherska ani oficjalna typowana tabela.

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

## Ograniczenia modelu

- Brak backtestu na sezonach historycznych — nie wiadomo, jak dobrze model przewidywałby np. 2025/26.
- Składy meczowe są założeniem; transfery, kontuzje i decyzje trenerskie w trakcie sezonu nie są modelowane.
- Ratingi zawodników są stałe przez cały sezon (bez formy, regresji do średniej ani rotacji).
- Parametry (`HTA`, szum toru, siła prioru `k`, μ pozycji, kara zmęczenia kevlaru) są dobrane ekspercko, nie estymowane z danych.
- Juniorzy i rezerwowi bez historii mocno zależą od prioru pozycji — duża niepewność właśnie w tych slotach.
- Statystyki z 1 ligi są skalowane (`×0.78`); to przybliżenie średniego spadku przy awansie, nie indywidualna kalibracja.
- Procenty z Monte Carlo wyglądają precyzyjnie, ale odzwierciedlają głównie siłę składu „na papierze” przy powyższych uproszczeniach.
