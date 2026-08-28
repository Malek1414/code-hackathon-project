# Camera-Based Basketball Score Tracker (long-term vision — NOT the hackathon build)

This is the roadmap product the hackathon demo points toward. Kept here as pitch ammunition.

Set up cameras at a basketball court that record the full playing area and automatically track the score — which team scored, and eventually which player scored. Long-term ambition: replace the whole scorer's table job (points, game clock, team fouls, timeouts, possession arrow) — one box combining what Pixellot/Veo (auto-filming), ShotTracker (chipped ball + tags) and Daktronics/Fair-Play (board + console) each do separately.

Key product decisions already made:
- **Auto-with-veto**: system calls baskets, human corrects on a tablet; corrections train the model.
- **Asymmetric errors**: never miss a real basket (fatal to trust); phantom points are tappable-away.
- **Uncertainty (2 vs 3)**: wall doesn't change; flashes "unconfirmed" on tablet.
- **Own display panel** rather than reverse-engineering closed Daktronics/Fair-Play consoles.
- **Referee clicker/mic** for clock start/stop (cameras can't see foul decisions).
- **Short video retention** (privacy — German gyms, minors).
- Target market: middle school, rec/summer leagues, AAU, German Landesliga/Oberliga/Regionalliga/NBBL — games run today by one harried volunteer or nobody.
- v1 points+clock → v2 per-player attribution → v3 rich stats (rebounds, assists, shot charts).

What has to be true (validation list): made-basket detection ≥~95%+, referees wear the clicker, correction loop genuinely improves the model, tripods survive a game, someone has a budget line, filming minors in German gyms is legally navigable.
