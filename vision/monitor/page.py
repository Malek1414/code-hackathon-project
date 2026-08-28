"""Single dark page. The browser polls /api/status every 3 s and re-renders;
images are only reloaded when their version token changes."""

HTML = r"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>FollowCam Monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #101216; --panel: #181b21; --line: #262a33; --ink: #e8e9ec; --muted: #8a909c;
    --blue: #7ad0ff; --green: #9be79b; --orange: #ffb86b; --red: #ff7b7b; --violet: #c8a4ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 13px/1.45 ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  header { display: flex; align-items: baseline; gap: 18px; padding: 12px 18px;
           border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--bg); z-index: 2; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .04em; }
  header .muted { color: var(--muted); }
  header .err { color: var(--red); }
  main { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; padding: 14px 18px 30px; }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; min-width: 0; }
  section.wide { grid-column: 1 / -1; }
  h2 { margin: 0 0 8px; font-size: 12px; letter-spacing: .12em; color: var(--muted); font-weight: 600; }
  h2 span { color: var(--ink); margin-left: 8px; letter-spacing: 0; font-weight: 400; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: 3px 14px; margin: 0 0 10px; }
  .kv dt { color: var(--muted); }
  .kv dd { margin: 0; overflow-wrap: anywhere; }
  .big { font-size: 22px; font-weight: 600; }
  .nothing { color: var(--muted); font-style: italic; }
  .error { color: var(--red); white-space: pre-wrap; }
  img.shot { width: 100%; height: auto; display: block; border-radius: 4px; border: 1px solid var(--line); background: #000; }
  table { border-collapse: collapse; width: 100%; margin-top: 6px; }
  th, td { text-align: left; padding: 3px 8px 3px 0; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; }
  td.num, th.num { text-align: right; }
  .made { color: var(--green); } .miss { color: var(--red); } .unconf { color: var(--orange); }
  .bar { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; margin: 4px 0 10px; }
  .bar i { display: block; height: 100%; background: var(--blue); }
  .t0 { color: var(--blue); } .t1 { color: var(--orange); } .tx { color: var(--muted); }
  .logs { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 10px; }
  .log { background: #0c0e12; border: 1px solid var(--line); border-radius: 4px; padding: 8px 10px; overflow-x: auto; }
  .log h3 { margin: 0 0 4px; font-size: 12px; font-weight: 500; color: var(--muted); }
  .log pre { margin: 0; font-size: 12px; line-height: 1.35; white-space: pre; color: #c9ccd3; }
  svg text { font: 11px ui-monospace, Menlo, monospace; fill: var(--muted); }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  a { color: var(--blue); }
  .on { color: var(--green); font-weight: 600; } .off { color: var(--muted); }
  .score { font-size: 26px; font-weight: 600; letter-spacing: .04em; }
</style>
</head>
<body>
<header>
  <h1>FOLLOWCAM PIPELINE</h1>
  <span class="muted">Stand <b id="now">…</b></span>
  <span class="muted">alle 3 s</span>
  <span id="conn" class="err"></span>
</header>
<main>
  <section id="label"><h2>LABEL</h2><div class="nothing">noch nichts</div></section>
  <section id="track"><h2>TRACK</h2><div class="nothing">noch nichts</div></section>
  <section id="numbers"><h2>NUMBERS</h2><div class="nothing">noch nichts</div></section>
  <section id="stats"><h2>STATS</h2><div class="nothing">noch nichts</div></section>
  <section id="court"><h2>COURT</h2><div class="nothing">noch nichts</div></section>
  <section id="live"><h2>LIVE</h2><div class="nothing">noch nichts</div></section>
  <section id="qa"><h2>QA</h2><div class="nothing">noch nichts</div></section>
  <section id="footage"><h2>FOOTAGE</h2><div class="nothing">noch nichts</div></section>
  <section id="logs" class="wide"><h2>LOGS</h2><div class="nothing">noch nichts</div></section>
</main>
<script>
const NOTHING = '<div class="nothing">noch nichts</div>';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d=0) => (v === null || v === undefined || Number.isNaN(v)) ? '?' : Number(v).toFixed(d);
const pct = v => (v === null || v === undefined) ? '?' : (100*v).toFixed(1) + ' %';
const teamCls = t => t === 0 || t === '0' ? 't0' : (t === 1 || t === '1' ? 't1' : 'tx');
const teamName = t => t === 0 || t === '0' ? 'Team A' : (t === 1 || t === '1' ? 'Team B' : 'unbekannt');
const errBox = e => `<div class="error">Fehler: ${esc(e)}</div>`;
const imgTokens = {};

function kv(rows) {
  return '<dl class="kv">' + rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('') + '</dl>';
}

function image(name, token, caption) {
  if (!token) return `<div class="nothing">${esc(caption)}: noch nichts</div>`;
  return `<img class="shot" id="img-${name}" data-token="${esc(token)}" alt="${esc(caption)}">`;
}

function syncImages() {
  document.querySelectorAll('img.shot').forEach(img => {
    const name = img.id.slice(4), token = img.dataset.token;
    if (imgTokens[name] !== token || !img.getAttribute('src')) {
      imgTokens[name] = token;
      img.src = `/img/${name}?v=${encodeURIComponent(token)}`;
    }
  });
}

function curveSvg(tr) {
  if (!tr || !tr.series || !tr.epochs) return '';
  const W = 620, H = 210, L = 44, R = 44, T = 14, B = 26;
  const xs = tr.epoch_values || [];
  const n = xs.length; if (n < 1) return '';
  const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);
  const x = e => L + (e - xmin) / (xmax - xmin) * (W - L - R);
  const colors = ['#9be79b', '#7ad0ff', '#ffb86b', '#c8a4ff', '#ff7b7b', '#e8e9ec'];
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Trainingskurve">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="#0c0e12" rx="4"/>`;
  const loss = tr.series.box_loss || [];
  const lmax = Math.max(0.01, ...loss.filter(v => v !== null));
  const yL = v => T + (1 - v / lmax) * (H - T - B);
  const yR = v => T + (1 - Math.min(1, Math.max(0, v))) * (H - T - B);
  [0, .25, .5, .75, 1].forEach(f => {
    const y = T + (1 - f) * (H - T - B);
    svg += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#262a33"/>`;
    svg += `<text x="${L - 6}" y="${y + 4}" text-anchor="end">${(lmax * f).toFixed(2)}</text>`;
    svg += `<text x="${W - R + 6}" y="${y + 4}">${(f).toFixed(2)}</text>`;
  });
  svg += `<text x="${L}" y="${H - 8}">Epoche ${xmin}</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${xmax}</text>`;
  svg += `<text x="4" y="10">box_loss</text><text x="${W - 4}" y="10" text-anchor="end">mAP50</text>`;
  const legend = [];
  const line = (vals, yfn, color, dash) => {
    const pts = vals.map((v, i) => v === null ? null : `${x(xs[i]).toFixed(1)},${yfn(v).toFixed(1)}`).filter(Boolean);
    if (pts.length) svg += `<polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="2" ${dash ? 'stroke-dasharray="4 3"' : ''}/>`;
  };
  if (loss.length) { line(loss, yL, '#ff7b7b', true); legend.push(['#ff7b7b', 'box_loss']); }
  let ci = 0;
  for (const [name, vals] of Object.entries(tr.series)) {
    if (name === 'box_loss') continue;
    const c = colors[ci++ % colors.length];
    line(vals, yR, c, false); legend.push([c, name]);
  }
  let lx = L + 6;
  legend.forEach(([c, name]) => {
    svg += `<rect x="${lx}" y="${T + 4}" width="10" height="3" fill="${c}"/><text x="${lx + 14}" y="${T + 9}">${esc(name)}</text>`;
    lx += 20 + name.length * 7;
  });
  return svg + '</svg>';
}

function renderLabel(d, images) {
  if (!d || d.error) return errBox(d && d.error || 'unbekannt');
  const cls = Object.entries(d.per_class || {});
  let html = kv([
    ['gelabelt', `<span class="big">${d.labels}</span> von ${d.frames} Frames` +
      (d.per_split && Object.keys(d.per_split).length ? ` <span class="tx">(${Object.entries(d.per_split).map(([k, v]) => `${esc(k)} ${v}`).join(', ')})</span>` : '')],
    ['Boxen', cls.length ? cls.map(([k, v]) => `${esc(k)} <b>${v}</b>`).join(' &nbsp; ') + ` &nbsp; <span class="tx">gesamt ${d.boxes_total}</span>` : '<span class="nothing">noch nichts</span>'],
    ['neueste', d.newest ? `${esc(d.newest)} <span class="tx">${esc(d.newest_time)}</span>` : '<span class="nothing">noch nichts</span>'],
    ['best.pt', d.best_pt && d.best_pt.exists ? `da, ${d.best_pt.mb} MB, ${esc(d.best_pt.time)}` : '<span class="nothing">noch nichts</span>'],
  ]);
  html += `<div class="bar"><i style="width:${d.frames ? Math.min(100, 100 * d.labels / d.frames) : 0}%"></i></div>`;
  html += image('label', images.label, 'neuestes Label');
  const tr = d.training;
  if (tr && tr.epochs) {
    const last = tr.last || {};
    html += `<h2 style="margin-top:12px">TRAINING <span>${esc(tr.run)} &nbsp; Epoche ${esc(last.epoch ?? tr.epochs)} &nbsp; box_loss ${esc(last['train/box_loss'] ?? '?')} &nbsp; mAP50 ${esc(last['metrics/mAP50(B)'] ?? '?')} &nbsp; <span class="tx">${esc(tr.mtime)}</span></span></h2>`;
    html += curveSvg(tr);
  } else {
    html += `<h2 style="margin-top:12px">TRAINING</h2>${NOTHING}`;
  }
  return html;
}

function renderTrack(d, images) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  if (!d.ok) return NOTHING;
  const tb = d.team_boxes || {}, ti = d.team_ids || {};
  let html = kv([
    ['tracks.jsonl', `<span class="big">${d.lines}</span> Zeilen &nbsp; <span class="tx">${d.file_mb} MB, ${esc(d.file_time)}</span>`],
    ['Fortschritt', d.progress_pct !== null && d.progress_pct !== undefined ? `${d.progress_pct} % &nbsp; <span class="tx">Frame ${d.last_frame} von ${esc(d.clip || '?')}, t = ${num(d.last_t, 1)} s</span>` : `Frame ${esc(d.last_frame)}, t = ${num(d.last_t, 1)} s`],
    ['Ball', `<b>${pct(d.ball_rate)}</b> Trefferquote &nbsp; <span class="tx">${d.ball_hits} von ${d.window} letzten Zeilen</span>`],
    ['Teams', `<span class="t0">A ${tb['0'] ?? 0} Boxen, ${ti['0'] ?? 0} IDs</span> &nbsp; <span class="t1">B ${tb['1'] ?? 0} Boxen, ${ti['1'] ?? 0} IDs</span> &nbsp; <span class="tx">unbekannt ${tb['-1'] ?? 0} Boxen, ${ti['-1'] ?? 0} IDs</span>`],
    ['pro Frame', `${num(d.players_per_frame, 1)} Spieler, ${num(d.hoops_per_frame, 2)} Körbe`],
    ['Gewichte', d.weights ? esc(Object.values(d.weights).join(' + ')) : '<span class="tx">?</span>'],
    ['overlay.mp4', d.overlay && d.overlay.exists ? `da, ${d.overlay.mb} MB, ${esc(d.overlay.time)}` : '<span class="nothing">noch nichts</span>'],
  ]);
  if (d.progress_pct !== null && d.progress_pct !== undefined) html += `<div class="bar"><i style="width:${Math.min(100, d.progress_pct)}%"></i></div>`;
  if (images.track) {
    html += image('track', images.track, 'letztes Overlay-Bild');
    if (d.latest_jpg && d.latest_jpg.exists) html += `<div class="tx">overlay_latest.jpg ${esc(d.latest_jpg.time)}</div>`;
  } else if (d.overlay && d.overlay.exists) {
    html += '<div class="nothing">overlay.mp4 wird noch geschrieben, noch kein lesbares Bild (TRACK kann out/overlay_latest.jpg liefern)</div>';
  } else {
    html += '<div class="nothing">letztes Overlay-Bild: noch nichts</div>';
  }
  return html;
}

function renderStats(d) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  if (!d.ok) return NOTHING;
  let html = '';
  const ev = d.events;
  if (ev) {
    const teams = Object.entries(ev.per_team || {}).map(([t, v]) => `<span class="${teamCls(t)}">${teamName(t)} ${v.fgm}/${v.fga}</span>`).join(' &nbsp; ');
    html += kv([
      ['Würfe', `<span class="big">${ev.count}</span> &nbsp; ${ev.made} getroffen &nbsp; <span class="tx">${esc(ev.clip || '')}, ${esc(ev.time)}</span>`],
      ['pro Team', teams || '<span class="tx">keine</span>'],
    ]);
    if (ev.shots.length) {
      html += '<table><tr><th class="num">t</th><th>Team</th><th class="num">Spieler</th><th>Ergebnis</th></tr>';
      ev.shots.forEach(s => {
        const res = s.made ? '<span class="made">Treffer</span>' : '<span class="miss">daneben</span>';
        html += `<tr><td class="num">${num(s.t, 1)} s</td><td class="${teamCls(s.team)}">${teamName(s.team)}</td><td class="num">${esc(s.player_id ?? '?')}</td><td>${res}${s.unconfirmed ? ' <span class="unconf">unbestätigt</span>' : ''}</td></tr>`;
      });
      html += '</table>';
    }
  } else {
    html += `<h2>EVENTS</h2>${NOTHING}`;
  }
  const st = d.stats;
  html += `<h2 style="margin-top:12px">STATS.JSON${st ? ` <span class="tx">${esc(st.time)}</span>` : ''}</h2>`;
  if (st) {
    if (st.teams.length) html += kv([['Teams', st.teams.map(t => `<span class="${teamCls(t.team)}">${teamName(t.team)} ${t.fgm}/${t.fga}</span>`).join(' &nbsp; ')]]);
    if (st.players.length) {
      html += '<table><tr><th class="num">ID</th><th>Team</th><th class="num">FGA</th><th class="num">FGM</th><th class="num">FG%</th><th class="num">Ballbesitz</th><th class="num">Distanz</th></tr>';
      st.players.forEach(p => {
        html += `<tr><td class="num">${esc(p.id)}</td><td class="${teamCls(p.team)}">${teamName(p.team)}</td><td class="num">${esc(p.fga ?? 0)}</td><td class="num">${esc(p.fgm ?? 0)}</td><td class="num">${p.fg_pct === null || p.fg_pct === undefined ? '?' : (100 * p.fg_pct).toFixed(0) + ' %'}</td><td class="num">${num(p.possession_s, 1)} s</td><td class="num">${p.distance_m === null || p.distance_m === undefined ? '?' : num(p.distance_m, 0) + ' m'}</td></tr>`;
      });
      html += '</table>';
    } else html += NOTHING;
  } else html += NOTHING;
  return html;
}

function renderCourt(d, images) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  const c = d.calib;
  let html = kv([
    ['court_calib.json', c ? `da &nbsp; <span class="tx">${esc(c.time)}, ${esc(c.clip || '')}</span>` : '<span class="nothing">noch nichts</span>'],
    ['Keyframes', c ? `<span class="big">${c.keyframes}</span>` : '<span class="nothing">noch nichts</span>'],
    ['Reproj-Fehler', c && c.reproj_err_px !== null && c.reproj_err_px !== undefined ? `<b>${num(c.reproj_err_px, 2)} px</b>` : '<span class="nothing">noch nichts</span>'],
    ['Punkte', c ? `${c.points}` + (c.court_m ? ` <span class="tx">Feld ${c.court_m.length} x ${c.court_m.width} m</span>` : '') : '<span class="nothing">noch nichts</span>'],
    ['minimap.mp4', d.minimap ? 'da' : '<span class="nothing">noch nichts</span>'],
    ['dashboard.html', d.dashboard ? 'da' : '<span class="nothing">noch nichts</span>'],
  ]);
  html += (d.preview && !images.court)
    ? '<div class="nothing">court_propagate_preview.mp4 wird noch geschrieben, noch kein lesbares Bild</div>'
    : image('court', images.court, 'court_propagate_preview.mp4');
  return html;
}

function renderLogs(d) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  if (!d.ok || !d.logs.length) return NOTHING;
  return '<div class="logs">' + d.logs.map(l =>
    `<div class="log"><h3>${esc(l.name)} <span>${esc(l.time)}, ${l.bytes} B</span></h3><pre>${l.lines.length ? esc(l.lines.join('\n')) : 'leer'}</pre></div>`
  ).join('') + '</div>';
}

function renderNumbers(d, images) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  if (!d.ok) return NOTHING + (images.numbers ? image('numbers', images.numbers, 'numbers_preview.jpg') : '');
  let html = kv([
    ['Tracks', `<span class="big">${d.tracks_numbered}</span> von ${d.tracks_total} mit Nummer &nbsp; <span class="tx">${esc(d.clip || '')}, ${esc(d.time)}</span>`],
    ['Spieler', `${d.players_numbered} mit Nummer, ${d.players_total} Keys gesamt`],
  ]);
  html += `<div class="bar"><i style="width:${d.tracks_total ? Math.min(100, 100 * d.tracks_numbered / d.tracks_total) : 0}%"></i></div>`;
  if (d.players.length) {
    html += '<table><tr><th>Key</th><th>Team</th><th class="num">Nummer</th><th class="num">Tracks</th><th class="num">Votes</th><th class="num">Reads</th><th class="num">von</th><th class="num">bis</th></tr>';
    d.players.forEach(p => {
      html += `<tr><td><b>${esc(p.key)}</b></td><td class="${teamCls(p.team)}">${teamName(p.team)}</td><td class="num">${p.number === null || p.number === undefined ? '?' : esc(p.number)}</td><td class="num">${p.tracks}</td><td class="num">${p.votes}</td><td class="num">${p.reads}</td><td class="num">${num(p.first_t, 1)} s</td><td class="num">${num(p.last_t, 1)} s</td></tr>`;
    });
    html += '</table>';
  }
  html += '<div style="margin-top:8px"></div>' + image('numbers', images.numbers, 'numbers_preview.jpg');
  return html;
}

function renderQa(d) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  if (!d.ok) return NOTHING;
  const kinds = Object.entries(d.kinds || {}).map(([k, v]) => `${esc(k)} ${v}`).join(', ');
  return kv([
    ['Sheets', `<span class="big">${d.sheets}</span> in out/qa` + (kinds ? ` <span class="tx">(${kinds})</span>` : '')],
    ['index.html', d.index ? `<a href="/qa/index.html" target="_blank" rel="noopener">out/qa/index.html öffnen</a> <span class="tx">${esc(d.index_time)}</span>` : '<span class="nothing">noch nichts</span>'],
    ['neuestes', d.newest ? `<a href="/qa/${encodeURIComponent(d.newest)}" target="_blank" rel="noopener">${esc(d.newest)}</a> <span class="tx">${esc(d.newest_time)}</span>` : '<span class="nothing">noch nichts</span>'],
  ]);
}

function renderLive(d) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  let html = kv([
    ['Prozess', d.running ? `<span class="on">läuft</span> <span class="tx">${esc((d.procs || []).map(p => p.split(' ')[0]).join(', '))}</span>` : '<span class="off">kein live.py Prozess</span>'],
  ]);
  const ev = d.events;
  if (ev) {
    const a = ev.teams['0'] || {}, b = ev.teams['1'] || {};
    html += `<div class="score"><span class="t0">A ${a.points ?? 0}</span> : <span class="t1">${b.points ?? 0} B</span></div>`;
    html += kv([
      ['FG', `<span class="t0">A ${a.fgm ?? 0}/${a.fga ?? 0}</span> &nbsp; <span class="t1">B ${b.fgm ?? 0}/${b.fga ?? 0}</span>` + (ev.unassigned ? ` &nbsp; <span class="unconf">${ev.unassigned} nicht zugeordnet</span>` : '')],
      ['Würfe', `${ev.shots}`],
      ['Frames', `${esc(ev.frames_processed ?? '?')} verarbeitet, ${esc(ev.frames_rendered ?? '?')} gerendert` + (ev.rtmp_frames ? `, ${ev.rtmp_frames} per RTMP` : '')],
      ['Quelle', `${esc(ev.clip || '?')} <span class="tx">${esc(ev.time)}</span>`],
    ]);
  } else {
    html += '<div class="nothing">live_events.json: noch nichts</div>';
  }
  return html;
}

function renderFootage(d) {
  if (!d) return NOTHING;
  if (d.error) return errBox(d.error);
  if (!d.ok) return NOTHING;
  const dur = s => s === undefined || s === null ? '?' : (s >= 3600 ? `${Math.floor(s / 3600)}:${String(Math.floor(s % 3600 / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}` : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);
  let html = '<table><tr><th>Clip</th><th class="num">Größe</th><th class="num">Dauer</th><th class="num">Format</th></tr>';
  d.clips.forEach(c => {
    html += `<tr><td>${esc(c.name)}</td><td class="num">${c.mb} MB</td><td class="num">${dur(c.duration_s)}</td><td class="num">${c.width ? `${c.width}x${c.height}, ${num(c.fps, 0)} fps` : '?'}</td></tr>`;
  });
  return html + '</table>';
}

function setSection(id, title, body) {
  const el = document.getElementById(id);
  el.innerHTML = `<h2>${title}</h2>` + body;
}

async function tick() {
  try {
    const r = await fetch('/api/status', {cache: 'no-store'});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const s = await r.json();
    const images = s.images || {};
    document.getElementById('now').textContent = s.now;
    document.getElementById('conn').textContent = '';
    setSection('label', 'LABEL', renderLabel(s.label, images));
    setSection('track', 'TRACK', renderTrack(s.track, images));
    setSection('stats', 'STATS', renderStats(s.stats));
    setSection('court', 'COURT', renderCourt(s.court, images));
    setSection('numbers', 'NUMBERS', renderNumbers(s.numbers, images));
    setSection('live', 'LIVE', renderLive(s.live));
    setSection('qa', 'QA', renderQa(s.qa));
    setSection('footage', 'FOOTAGE', renderFootage(s.footage));
    setSection('logs', 'LOGS', renderLogs(s.logs));
    syncImages();
  } catch (e) {
    document.getElementById('conn').textContent = 'keine Verbindung: ' + e.message;
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""
