/* Lectern front end.
 *
 * Performance shape that matters on an 8 GB M1:
 *  - Sentences are the DOM unit (~400-600 <p> nodes for a 50-min lecture).
 *    Rendering 9,000 permanent word spans would multiply layout objects ~15x.
 *  - Word spans are materialised lazily, only for sentences that are hot
 *    (near the playhead) or hovered, and torn down again once cold.
 *  - The karaoke loop binary-searches an Int32Array and touches the DOM only
 *    when the active word actually changes (~2-3 times a second).
 *  - Auto-scroll uses an IntersectionObserver. Calling getBoundingClientRect()
 *    inside the paint loop would force a synchronous layout on every word.
 */
const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw Object.assign(new Error((body && body.error) || r.statusText), { body, status: r.status });
  return body;
};
const jpost = (p, o) => api(p, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(o || {}) });

const bytesText = (b) => b >= 1e9 ? `${(b / 1e9).toFixed(b >= 1e10 ? 0 : 1)} GB`
  : b >= 1e6 ? `${Math.round(b / 1e6)} MB` : b > 0 ? `${Math.max(1, Math.round(b / 1e3))} KB` : "0 KB";
const fmt = (s) => {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}`
           : `${m}:${String(x).padStart(2, "0")}`;
};
const clock = (s) => { s = Math.floor(s || 0);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; };

let toastTimer;
function toast(msg, ms = 3600) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), ms);
}

/* ------------------------------------------------------------- reader */
const R = {
  id: null, doc: null, t0: null, t1: null, sw0: null,
  els: [], hot: new Set(), visible: new Set(), io: null,
  activeWord: -1, activeSent: -1, raf: 0, follow: true,
};

function buildIndex(doc) {
  const w = doc.words;
  R.doc = doc;
  R.t0 = Int32Array.from(w.t0);
  R.t1 = Int32Array.from(w.t1);
  R.sw0 = Int32Array.from(doc.sents.map((s) => s.w0));
}

// Largest i with arr[i] <= v, or -1.
function lowerIdx(arr, v) {
  let lo = 0, hi = arr.length - 1, best = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1;
    if (arr[mid] <= v) { best = mid; lo = mid + 1; } else hi = mid - 1; }
  return best;
}
const wordAt = (ms) => lowerIdx(R.t0, ms);
const sentOfWord = (i) => (i < 0 ? -1 : Math.max(0, lowerIdx(R.sw0, i)));

function renderTranscript(doc) {
  const host = $("transcript");
  host.textContent = "";
  R.els = []; R.hot.clear(); R.visible.clear();
  R.activeWord = -1; R.activeSent = -1;
  if (R.io) R.io.disconnect();

  const words = doc.words.w;
  const frag = document.createDocumentFragment();
  doc.sents.forEach((s, si) => {
    const p = document.createElement("p");
    p.className = "s"; p.dataset.si = si;
    const ts = document.createElement("span");
    ts.className = "ts"; ts.textContent = clock(s.t0 / 1000);
    p.appendChild(ts);
    p.appendChild(document.createTextNode(
      words.slice(s.w0, s.w0 + s.wn).join(" ")));
    frag.appendChild(p);
    R.els.push(p);
  });
  host.appendChild(frag);

  // Visibility tracked passively; no forced layout in the paint loop.
  R.io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const si = +e.target.dataset.si;
      if (e.isIntersecting) R.visible.add(si); else R.visible.delete(si);
    }
  }, { root: host.closest(".pane"), threshold: 0.01 });
  R.els.forEach((el) => R.io.observe(el));
}

function materialize(si) {
  if (si < 0 || si >= R.els.length || R.hot.has(si)) return;
  const el = R.els[si], s = R.doc.sents[si], words = R.doc.words.w;
  const frag = document.createDocumentFragment();
  const ts = document.createElement("span");
  ts.className = "ts"; ts.textContent = clock(s.t0 / 1000);
  frag.appendChild(ts);
  for (let k = 0; k < s.wn; k++) {
    const gi = s.w0 + k;
    const sp = document.createElement("span");
    sp.className = "w"; sp.dataset.wi = gi; sp.textContent = words[gi];
    frag.appendChild(sp);
    if (k < s.wn - 1) frag.appendChild(document.createTextNode(" "));
  }
  el.textContent = ""; el.appendChild(frag);
  R.hot.add(si);
}

function dematerialize(si) {
  if (!R.hot.has(si)) return;
  const el = R.els[si], s = R.doc.sents[si];
  el.textContent = "";
  const ts = document.createElement("span");
  ts.className = "ts"; ts.textContent = clock(s.t0 / 1000);
  el.appendChild(ts);
  el.appendChild(document.createTextNode(
    R.doc.words.w.slice(s.w0, s.w0 + s.wn).join(" ")));
  R.hot.delete(si);
}

function keepHot(centre) {
  for (let d = -1; d <= 1; d++) materialize(centre + d);
  if (R.hot.size > 10) {
    for (const si of Array.from(R.hot)) {
      if (Math.abs(si - centre) > 3 && si !== R.activeSent) dematerialize(si);
    }
  }
}

// Scroll only the transcript pane (scrollIntoView would also scroll every
// overflow:hidden ancestor, pushing the header off-screen).
function centreInPane(el, behavior) {
  const pane = $("transcript").closest(".pane");
  const pr = pane.getBoundingClientRect(), er = el.getBoundingClientRect();
  pane.scrollTo({ top: pane.scrollTop + (er.top - pr.top) - pane.clientHeight / 2 + er.height / 2, behavior });
}

function setFollow(on) {
  R.follow = on;
  $("followChk").checked = on;
  $("jumpNow").classList.toggle("hidden", on);
}

function paint(ms) {
  const wi = wordAt(ms);
  if (wi === R.activeWord) return;
  R.activeWord = wi;
  const si = sentOfWord(wi);

  if (si !== R.activeSent) {
    if (R.activeSent >= 0 && R.els[R.activeSent]) R.els[R.activeSent].classList.remove("active");
    R.activeSent = si;
    if (si >= 0 && R.els[si]) {
      R.els[si].classList.add("active");
      keepHot(si);
      if (R.follow && !R.visible.has(si)) {
        centreInPane(R.els[si], "smooth");
      }
    }
  } else {
    keepHot(si);
  }

  const host = $("transcript");
  const prev = host.querySelector(".w.on");
  if (prev) prev.classList.remove("on");
  if (wi >= 0) {
    const cur = host.querySelector(`.w[data-wi="${wi}"]`);
    if (cur) cur.classList.add("on");
  }
}

/* -------------------------------------------------------------- audio */
const audio = $("audio");

// Land a beat before the word rather than exactly on its first phoneme --
// otherwise the very sound you clicked for is half-missed. Seeking itself is
// sample-exact; this is a deliberate, small lead-in.
const PREROLL_MS = 150;

function seekMs(ms, { preroll = true } = {}) {
  if (!Number.isFinite(ms)) return;
  audio.currentTime = Math.max(0, (ms - (preroll ? PREROLL_MS : 0)) / 1000);
  if (audio.paused) audio.play().catch(() => {});
}

audio.addEventListener("timeupdate", () => {
  // Also repaint here, not only from rAF: requestAnimationFrame is throttled to
  // a standstill when the tab is backgrounded, while timeupdate keeps firing at
  // ~4 Hz. paint() early-returns when the word has not changed, so the extra
  // call is free.
  if (R.doc) paint(audio.currentTime * 1000);
  const d = audio.duration || (R.doc ? R.doc.duration_ms / 1000 : 0);
  $("curTime").textContent = fmt(audio.currentTime);
  if (d && Number.isFinite(d)) {
    $("durTime").textContent = fmt(d);
    $("seek").value = String(Math.round((audio.currentTime / d) * 1000));
  }
});
audio.addEventListener("play", () => { $("playBtn").textContent = "❚❚"; loop(); });
audio.addEventListener("pause", () => { $("playBtn").textContent = "▶"; cancelAnimationFrame(R.raf); });
audio.addEventListener("ended", () => { $("playBtn").textContent = "▶"; });
audio.addEventListener("seeked", () => { if (R.doc) paint(audio.currentTime * 1000); });

function loop() {
  cancelAnimationFrame(R.raf);
  const tick = () => {
    if (R.doc && !audio.paused) paint(audio.currentTime * 1000);
    if (!audio.paused) R.raf = requestAnimationFrame(tick);
  };
  R.raf = requestAnimationFrame(tick);
}

$("playBtn").onclick = () => (audio.paused ? audio.play().catch(() => {}) : audio.pause());
$("seek").oninput = (e) => {
  const d = audio.duration || (R.doc ? R.doc.duration_ms / 1000 : 0);
  if (d && Number.isFinite(d)) audio.currentTime = (e.target.value / 1000) * d;
};
// Playback speed, 0.5x-2x. Remembered across lectures and restarts; set as the
// default rate too, because loading a new src resets playbackRate to it.
const RATES = [...$("rate").options].map((o) => parseFloat(o.value));
function setRate(r, announce) {
  r = RATES.includes(r) ? r : 1;
  audio.defaultPlaybackRate = audio.playbackRate = r;
  audio.preservesPitch = true;               // faster, not chipmunk
  $("rate").value = String(r);
  $("rate").classList.toggle("off1", r !== 1);
  try { localStorage.setItem("lectern.rate", String(r)); } catch { /* private mode */ }
  if (announce) toast(`Speed ${r}×`, 900);
}
function stepRate(dir) {
  const i = RATES.indexOf(audio.playbackRate);
  setRate(RATES[Math.max(0, Math.min(RATES.length - 1, (i < 0 ? RATES.indexOf(1) : i) + dir))], true);
}
$("rate").onchange = (e) => setRate(parseFloat(e.target.value));
audio.addEventListener("loadedmetadata", () => { audio.playbackRate = audio.defaultPlaybackRate; });
setRate(parseFloat((() => { try { return localStorage.getItem("lectern.rate"); } catch { return null; } })() || "1"));
$("followChk").onchange = (e) => setFollow(e.target.checked);

// Taking over the scroll bar means "stop following": otherwise the page would
// drag you back to the playhead every few seconds while you read ahead.
{
  const pane = $("transcript").closest(".pane");
  const manual = () => { if (R.follow) setFollow(false); };
  pane.addEventListener("wheel", manual, { passive: true });
  pane.addEventListener("touchmove", manual, { passive: true });
  pane.addEventListener("keydown", (e) => {
    if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(e.key)) manual();
  });
}
$("jumpNow").onclick = () => {
  setFollow(true);
  if (R.activeSent >= 0 && R.els[R.activeSent]) centreInPane(R.els[R.activeSent], "smooth");
};

// Notes | Transcript tabs (narrow windows and phones only; hidden by CSS otherwise).
$("paneTabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-tab]");
  if (!b) return;
  $("panes").dataset.tab = b.dataset.tab;
  for (const x of $("paneTabs").children) x.classList.toggle("on", x === b);
});

// Click-to-jump. Delegated: one listener for the whole transcript.
$("transcript").addEventListener("click", (e) => {
  const w = e.target.closest(".w");
  if (w) { seekMs(R.t0[+w.dataset.wi]); return; }
  const s = e.target.closest(".s");
  if (s) seekMs(R.doc.sents[+s.dataset.si].t0);
});
// Hovering makes a sentence word-clickable without paying for it up front.
$("transcript").addEventListener("mouseover", (e) => {
  const s = e.target.closest(".s");
  if (s) materialize(+s.dataset.si);
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input,select,textarea")) return;
  if ($("reader").classList.contains("hidden")) return;
  if (e.code === "Space") { e.preventDefault(); $("playBtn").click(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); audio.currentTime -= e.shiftKey ? 30 : 5; }
  else if (e.key === "ArrowRight") { e.preventDefault(); audio.currentTime += e.shiftKey ? 30 : 5; }
  else if (e.key === "[") { e.preventDefault(); stepRate(-1); }
  else if (e.key === "]") { e.preventDefault(); stepRate(+1); }
  else if (e.key === "\\") { e.preventDefault(); setRate(1, true); }
});

/* -------------------------------------------------------------- notes */
function noteButton(b) {
  const el = document.createElement("button");
  el.className = "nb" + (b.tier === "region" ? " approx" : "") +
                 (b.grounded !== undefined && b.grounded < 0.5 ? " unverified" : "");
  const t = document.createElement("span");
  t.className = "nt";
  t.textContent = clock(b.t_ms / 1000);
  const txt = document.createElement("span");
  txt.textContent = b.text;
  el.append(t, txt);
  el.title = b.tier === "region"
    ? "Approximate — this note summarises a stretch of the lecture. Jumps to about here."
    : (b.preview ? `Jumps to: “${b.preview}”` : "Jumps to this moment");
  if (b.grounded !== undefined && b.grounded < 0.5) {
    el.title += "\nSome wording here does not appear in the lecture — double-check it.";
  }
  el.onclick = () => seekMs(b.t_ms);
  return el;
}

function renderNotes(doc) {
  const host = $("notes");
  host.textContent = "";
  if (doc.fallback) {
    const w = document.createElement("p");
    w.className = "note-warn";
    w.textContent = (doc.sections || []).length
      ? "Written without the language model (" + doc.fallback +
        "). These are the most important sentences taken straight from the transcript."
      : "No study notes for this recording: " + doc.fallback + ".";
    host.appendChild(w);
  }
  if (doc.summary) {
    const p = document.createElement("p");
    p.className = "note-sum";
    p.textContent = doc.summary;
    host.appendChild(p);
  }
  for (const sec of doc.sections || []) {
    const h = document.createElement("h4");
    h.className = "nsec";
    h.textContent = sec.heading;
    host.appendChild(h);
    const ul = document.createElement("ul");
    ul.className = "nlist";
    for (const b of sec.bullets) {
      const li = document.createElement("li");
      li.appendChild(noteButton(b));
      ul.appendChild(li);
    }
    host.appendChild(ul);
  }
  if ((doc.terms || []).length) {
    const h = document.createElement("h4");
    h.className = "nsec"; h.textContent = "Key terms";
    host.appendChild(h);
    const ul = document.createElement("ul");
    ul.className = "nlist";
    for (const t of doc.terms) {
      const li = document.createElement("li");
      const b = noteButton({ ...t, text: "" });
      b.querySelector("span:last-child").innerHTML =
        `<b>${esc(t.term)}</b> — ${esc(t.definition)}`;
      li.appendChild(b);
      ul.appendChild(li);
    }
    host.appendChild(ul);
  }
  if ((doc.quiz || []).length) {
    const det = document.createElement("details");
    det.className = "quiz";
    det.innerHTML = `<summary>Check yourself (${doc.quiz.length})</summary>`;
    for (const q of doc.quiz) {
      const d2 = document.createElement("details");
      d2.className = "qa";
      d2.innerHTML = `<summary></summary><p></p>`;
      d2.querySelector("summary").textContent = q.q;
      d2.querySelector("p").textContent = q.a;
      det.appendChild(d2);
    }
    host.appendChild(det);
  }
  const foot = document.createElement("p");
  foot.className = "note-foot";
  foot.innerHTML = `Written on this Mac by <code>${esc(doc.engine || "?")}</code>. ` +
    `Dimmed timestamps are approximate. ` +
    `<button id="regenNotes" class="ghost tiny">Rewrite</button>`;
  host.appendChild(foot);
  $("regenNotes").onclick = async (e) => {
    e.target.disabled = true; e.target.textContent = "working…";
    await jpost(`/api/lecture/${encodeURIComponent(R.id)}/notes/regenerate`);
    refreshLecture(false);
  };
}

/* ------------------------------------------------------------ routing */
function showLibrary() {
  $("reader").classList.add("hidden");
  $("library").classList.remove("hidden");
  $("backBtn").classList.add("hidden");
  $("player").classList.add("hidden");
  $("appTitle").textContent = "Lectern";
  audio.pause(); audio.removeAttribute("src"); audio.load();
  R.id = null; R.doc = null;
  location.hash = "";
  loadLibrary();
}

async function openLecture(id) {
  R.id = id;
  $("library").classList.add("hidden");
  $("reader").classList.remove("hidden");
  $("backBtn").classList.remove("hidden");
  location.hash = `#/l/${id}`;
  $("transcript").textContent = "";
  $("lecTitle").textContent = "…"; $("lecMeta").textContent = "";
  await refreshLecture(true);
}

async function refreshLecture(first) {
  if (!R.id) return;
  let d;
  try { d = await api(`/api/lecture/${encodeURIComponent(R.id)}`); }
  catch { toast("Could not load that lecture."); return showLibrary(); }

  R.lastLecture = d;
  $("lecTitle").textContent = d.title;
  $("appTitle").textContent = d.folder || d.course || "Lectern";
  const bits = [];
  if (d.created_utc) bits.push(new Date(d.created_utc).toLocaleString());
  if (d.duration_ms) bits.push(fmt(d.duration_ms / 1000));
  if (d.words) bits.push(`${d.words.toLocaleString()} words`);
  if (d.bytes) bits.push(`${(d.bytes / 1e6).toFixed(1)} MB`);
  $("lecMeta").textContent = bits.join(" · ");

  await renderClassRow(d);
  if (!(d.job && !d.job.done)) {           // the job loop already refreshes
    recheckWhilePending("lecture", d.pending_folder != null, () => refreshLecture(false));
  }

  const ban = $("lecBanner");
  if (d.recovery_note) { ban.textContent = d.recovery_note; ban.classList.remove("hidden"); }
  else ban.classList.add("hidden");

  // A lecture is resumable when it is explicitly paused, or fully finished
  // and idle — never while it is actively processing, and never while a
  // DIFFERENT lecture is being recorded right now.
  const resumable = ["paused", "complete", "failed"].includes(d.state)
    && !(d.job && !d.job.done) && !recActive;
  const rb = $("resumeBox");
  if (resumable) {
    rb.classList.remove("hidden");
    $("resumeText").textContent = d.state === "paused"
      ? pausedText(d)
      : "Missed something? You can record more into this same lecture.";
    $("resumeBtn").textContent = d.state === "paused" ? "Continue recording" : "Record more";
    // A paused lecture can be finished without recording more.
    $("finishBtn").classList.toggle("hidden", d.state !== "paused");
  } else {
    rb.classList.add("hidden");
  }

  const job = d.job;
  const box = $("jobBox");
  if (job && !job.done) {
    box.classList.remove("hidden");
    $("jobStage").textContent = job.message || job.stage;
    $("jobPct").textContent = job.percent ? `${job.percent}%` : "";
    $("jobFill").style.width = `${job.percent || 0}%`;
    $("jobHint").textContent =
      "This runs in the background on this computer. You can leave this page.";
    setTimeout(() => refreshLecture(false), 1200);
  } else if (job && job.error) {
    box.classList.remove("hidden");
    $("jobStage").textContent = "Processing failed";
    $("jobPct").textContent = "";
    $("jobFill").style.width = "0%";
    $("jobHint").textContent = job.error;
  } else {
    box.classList.add("hidden");
  }

  if (d.has_audio) {
    $("player").classList.remove("hidden");
    const src = `/api/lecture/${encodeURIComponent(R.id)}/audio`;
    if (!audio.src || !audio.src.endsWith(src)) audio.src = src;
  } else $("player").classList.add("hidden");

  if (d.has_notes) {
    try { renderNotes(await api(`/api/lecture/${encodeURIComponent(R.id)}/notes`)); }
    catch { /* not ready */ }
  } else if (!job) {
    $("notes").innerHTML =
      `<p class="muted">No notes yet. ` +
      `<button id="genNotes" class="ghost">Write study notes</button></p>`;
    const g = $("genNotes");
    if (g) g.onclick = async () => {
      g.disabled = true; g.textContent = "starting…";
      await jpost(`/api/lecture/${encodeURIComponent(R.id)}/notes/regenerate`);
      refreshLecture(false);
    };
  }

  if (d.has_transcript && (!R.doc || first)) {
    try {
      const doc = await api(`/api/lecture/${encodeURIComponent(R.id)}/transcript`);
      buildIndex(doc); renderTranscript(doc);
      $("durTime").textContent = fmt(doc.duration_ms / 1000);
    } catch { /* still processing */ }
  } else if (!d.has_transcript && !job) {
    $("transcript").innerHTML =
      `<p class="muted">No transcript yet. <button id="reproc" class="ghost">Process now</button></p>`;
    const b = $("reproc");
    if (b) b.onclick = async () => {
      await jpost(`/api/lecture/${encodeURIComponent(R.id)}/process`);
      refreshLecture(false);
    };
  }
}

/* ------------------------------------------------------------ folders */
const L = { folders: [], lectures: [], sel: localStorage.getItem("lectern.folder") ?? "\u0000" };
// sel is "\u0000" for "All", "" for Unfiled, otherwise the folder name.
const ALL = "\u0000";

async function loadFolders() {
  try { L.folders = (await api("/api/folders")).folders; } catch { L.folders = []; }
  const ul = $("folderList");
  ul.textContent = "";
  const rows = [{ key: ALL, label: "All lectures", icon: "\u25e6",
                  count: L.lectures.length }];
  for (const f of L.folders) {
    if (f.name === "") continue;
    rows.push({ key: f.name, label: f.name, icon: "\u25b8", count: f.count });
  }
  const unfiled = L.lectures.filter((x) => !x.folder).length;
  if (unfiled) rows.push({ key: "", label: "Unfiled", icon: "\u00b7", count: unfiled });

  for (const r of rows) {
    const li = document.createElement("li");
    if (r.key === L.sel) li.className = "sel";
    li.innerHTML = `<span class="ico">${r.icon}</span><span class="lbl"></span>` +
                   `<span class="n">${r.count}</span>`;
    li.querySelector(".lbl").textContent = r.label;
    li.onclick = () => {
      L.sel = r.key;
      localStorage.setItem("lectern.folder", r.key);
      for (const c of ul.children) c.classList.remove("sel");
      li.classList.add("sel");
      renderLibrary();
    };
    if (r.key !== ALL && r.key !== "") {
      li.ondblclick = () => renameFolder(r.key);
      li.title = "Double-click to rename";
    }
    ul.appendChild(li);
  }
}

const NEW_CLASS = "\u0001new";
// Same rules as storage.clean_folder, so the name shown and remembered is the
// name of the directory the server actually creates.
const cleanFolder = (s) => {
  const n = String(s || "").trim().replace(/[\/\\:\x00-\x1f]/g, "").replace(/^[. ]+|[. ]+$/g, "");
  // 60 characters, not UTF-16 units: slicing an emoji in half sends the
  // server a lone surrogate, which no filesystem name can hold.
  return n === "." || n === ".." ? "" : Array.from(n).slice(0, 60).join("").trim();
};
const folderNames = () => L.folders.map((f) => f.name).filter(Boolean);

function folderOptions(selected, { withNew = false } = {}) {
  const opts = [`<option value=""${selected ? "" : " selected"}>Unfiled</option>`];
  // A queued move to a class that does not exist yet (made with "New class")
  // still has to show as the selected value.
  const names = folderNames();
  if (selected && selected !== NEW_CLASS && !names.includes(selected)) names.push(selected);
  for (const n of names) {
    opts.push(`<option value="${esc(n)}"${n === selected ? " selected" : ""}>${esc(n)}</option>`);
  }
  if (withNew) opts.push(`<option value="${NEW_CLASS}">\uff0b New class\u2026</option>`);
  return opts.join("");
}

async function refreshFolders() {
  try { L.folders = (await api("/api/folders")).folders; } catch { /* keep what we have */ }
}

// "biology 102126" -> BIO, "hist101226" -> History, "chem092826" -> chem.
// Only a suggestion: the dialog shows where the lecture will go, and any
// choice the user makes themselves is never overridden.
const GUESS_SKIP = new Set(["the", "and", "for", "with", "class", "lecture", "lec",
  "notes", "part", "week", "day", "review", "exam", "quiz", "test", "final", "intro"]);
function guessFolder(title) {
  const toks = ((title || "").toLowerCase().match(/[a-z]+/g) || [])
    .filter((t) => t.length >= 3 && !GUESS_SKIP.has(t));
  let best = null, bestLen = 0;
  for (const n of folderNames()) {
    const f = n.toLowerCase().replace(/[^a-z]/g, "");
    if (f.length < 3) continue;
    for (const t of toks) {
      if (t.startsWith(f) || f.startsWith(t)) {
        const len = Math.min(t.length, f.length);
        if (len > bestLen) { best = n; bestLen = len; }
      }
    }
  }
  return best;
}

async function moveLecture(id, folder) {
  try {
    const r = await jpost(`/api/lecture/${encodeURIComponent(id)}/folder`, { folder });
    // Names as the server stored them. A deferred move back to where it
    // already is just cancels the queued one (pending_folder comes back null).
    const here = r.folder || "Unfiled";
    if (!r.deferred) toast(`Moved to ${here}.`, 3600);
    else if (r.pending_folder == null) toast(`Staying in ${here}.`, 3600);
    else toast(`Will move to ${r.pending_folder || "Unfiled"} as soon as it finishes recording or processing.`, 5200);
    await loadLibrary();
    if (R.id === id) refreshLecture(false);
    return true;
  } catch (e) { toast(e.message || "Could not move that lecture."); return false; }
}

async function askNewClass() {
  return cleanFolder(prompt("Name of the new class (e.g. BIO 101)")) || null;
}

// Cancel is type="button" in these dialogs so that Enter in a text field
// submits the form (Start recording / Create) instead of hitting Cancel.
document.querySelectorAll("dialog [data-close]").forEach((b) => {
  b.onclick = () => b.closest("dialog").close("cancel");
});

// Where a lecture is, or is about to be. pending_folder is null when no move
// is queued, "" when it is queued for Unfiled.
const effFolder = (l) => (l.pending_folder != null ? l.pending_folder : (l.folder || ""));

function openMoveMenu(anchor, l) {
  document.querySelectorAll(".move-menu,.card-menu").forEach((m) => m.remove());
  const menu = document.createElement("div");
  menu.className = "move-menu";
  const cur = effFolder(l);
  const rows = ["", ...folderNames()];
  menu.innerHTML = `<div class="mm-head">Move to class</div>` + rows.map((n, i) =>
    `<button data-i="${i}"><span class="ck">${n === cur ? "\u2713" : ""}</span>${esc(n || "Unfiled")}</button>`
  ).join("") + `<hr><button data-new><span class="ck"></span>\uff0b New class\u2026</button>`;
  menu.querySelectorAll("[data-i]").forEach((b) => {
    b.onclick = (e) => {
      e.stopPropagation(); menu.remove();
      const target = rows[+b.dataset.i];
      if (target !== cur) moveLecture(l.id, target);
    };
  });
  menu.querySelector("[data-new]").onclick = async (e) => {
    e.stopPropagation(); menu.remove();
    const name = await askNewClass();
    if (name) moveLecture(l.id, name);
  };
  anchor.appendChild(menu);
  const close = (e) => {
    if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener("click", close); }
  };
  setTimeout(() => document.addEventListener("click", close), 0);
}

// The lecture page's "Class" dropdown.
async function renderClassRow(d) {
  if (!L.folders.length) await refreshFolders();
  const sel = $("lecFolder");
  const cur = effFolder(d);
  // refreshLecture re-runs every ~1 s while processing; rebuilding an open
  // <select> would snap it shut under the user's cursor.
  const sig = JSON.stringify([folderNames(), cur]);
  if (sel.dataset.sig !== sig || document.activeElement !== sel) {
    sel.innerHTML = folderOptions(cur, { withNew: true });
    sel.value = cur;
    sel.dataset.sig = sig;
  }
  const note = $("lecFolderNote");
  note.textContent = d.pending_folder != null
    ? `Moves to ${d.pending_folder || "Unfiled"} as soon as it finishes recording or processing.`
    : "";
  sel.onchange = async () => {
    let target = sel.value;
    if (target === NEW_CLASS) {
      target = await askNewClass();
      if (target == null) { sel.value = cur; return; }
    }
    if (target === cur) { sel.value = cur; return; }
    if (await moveLecture(d.id, target)) delete sel.dataset.sig;
    else sel.value = cur;
  };
}

function pausedText(d) {
  const at = d.paused_utc
    ? ` at ${new Date(d.paused_utc).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`
    : "";
  switch (d.pause_reason) {
    case "lid closed":
      return `Paused${at} because the laptop lid was closed. The mic is off and everything up to then is saved.`;
    case "system sleep":
      return `Paused${at} because the Mac went to sleep. The mic is off and everything up to then is saved.`;
    case "recorder stopped unexpectedly":
      return `Recording stopped unexpectedly${at}. Everything up to then is saved.`;
    case "disk full":
      return `Paused${at} because this Mac is almost out of disk space. Everything up to then is saved — free some space to continue.`;
    default:
      return "This recording is paused. Nothing has been lost.";
  }
}

$("newFolderBtn").onclick = () => {
  $("folderDlgTitle").textContent = "New class folder";
  $("fFolderName").value = "";
  $("folderForm").dataset.mode = "create";
  $("folderDlg").showModal();
};

function renameFolder(name) {
  $("folderDlgTitle").textContent = "Rename class folder";
  $("fFolderName").value = name;
  $("folderForm").dataset.mode = "rename";
  $("folderForm").dataset.old = name;
  $("folderDlg").showModal();
}

$("folderForm").addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value !== "go") return;
  const name = $("fFolderName").value.trim();
  if (!name) return;
  const form = $("folderForm");
  try {
    if (form.dataset.mode === "rename") {
      const old = form.dataset.old;
      // The server cleans the name ("MATH: 101" -> "MATH 101"); use what it
      // stored, or the sidebar selection would point at a class that does not exist.
      const r = await api(`/api/folders/${encodeURIComponent(old)}`, {
        method: "PATCH", headers: { "content-type": "application/json" },
        body: JSON.stringify({ name }) });
      if (L.sel === old) { L.sel = r.name; localStorage.setItem("lectern.folder", r.name); }
      toast(`Renamed to ${r.name}.`);
    } else {
      const r = await jpost("/api/folders", { name });
      L.sel = r.name; localStorage.setItem("lectern.folder", r.name);
      toast(`Created ${r.name}.`);
    }
    await loadLibrary();
  } catch (err) { toast(err.message || "That did not work."); }
});

/* ------------------------------------------------------------ library */
// A queued class move lands a few seconds after the lecture goes idle; keep
// re-checking until it has, so the card and the counts never stay stale.
// One timer per view, so the lecture page and the library never cancel each
// other's re-check.
const pendingTimers = {};
function recheckWhilePending(view, pending, fn) {
  clearTimeout(pendingTimers[view]);
  if (pending) pendingTimers[view] = setTimeout(fn, 3500);
}

let libSig = "";
async function loadLibrary() {
  try { L.lectures = (await api("/api/lectures")).lectures; } catch { return; }
  await loadFolders();
  // Re-rendering replaces every card, which would close a Move or ⋯ menu the
  // user has open; skip it then (the next re-check catches up), and skip it
  // when nothing changed.
  const menuOpen = document.querySelector("#lectureList .move-menu, #lectureList .card-menu");
  if (!menuOpen && JSON.stringify([L.lectures, L.sel]) !== libSig) renderLibrary();
  recheckWhilePending("library",
    !$("library").classList.contains("hidden")
      && (menuOpen || L.lectures.some((l) => l.pending_folder != null)),
    loadLibrary);
  try {
    const s = await api("/api/status");
    const used = L.lectures.reduce((a, x) => a + (x.bytes || 0), 0);
    const size = used >= 1e9 ? `${(used / 1e9).toFixed(1)} GB`
               : used >= 1e6 ? `${(used / 1e6).toFixed(used < 1e7 ? 1 : 0)} MB`
               : `${Math.max(1, Math.round(used / 1e3))} KB`;
    let appLine = `${size} used`;
    try {
      appLine = `Lectern uses ${bytesText((await api("/api/storage")).total_bytes)}`;
    } catch { /* keep the lecture-only figure */ }
    $("storageLine").textContent =
      `${L.lectures.length} lecture${L.lectures.length === 1 ? "" : "s"} \u00b7 ${appLine}\n` +
      `${(s.free_bytes / 1e9).toFixed(0)} GB free on this Mac`;
  } catch { /* non-critical */ }
}

function renderLibrary() {
  libSig = JSON.stringify([L.lectures, L.sel]);
  const ul = $("lectureList");
  ul.textContent = "";
  const list = L.lectures.filter((l) =>
    L.sel === ALL ? true : (l.folder || "") === L.sel);
  $("libEmpty").classList.toggle("hidden", list.length > 0);
  if (!list.length && L.sel !== ALL) {
    $("libEmpty").classList.remove("hidden");
    $("libEmpty").querySelector("h2").textContent = "Nothing here yet";
    $("libEmpty").querySelector("p").innerHTML =
      "No lectures filed under <b>" + esc(L.sel || "Unfiled") + "</b>.";
  } else if (!list.length) {
    $("libEmpty").querySelector("h2").textContent = "No lectures yet";
    $("libEmpty").querySelector("p").innerHTML =
      "Hit <b>Record a lecture</b> when class starts. Everything stays on this Mac.";
  }
  for (const l of list) {
    const li = document.createElement("li");
    li.className = "card";
    const busy = ["importing", "concatenating", "transcribing", "noting", "archiving"].includes(l.state);
    li.innerHTML = `
      <div style="min-width:0">
        <div class="t1">${esc(l.title)}${
          busy ? '<span class="badge work">processing</span>' : ""}${
          l.state === "paused" ? `<span class="badge warn"${l.pause_reason ? ` title="${esc(l.pause_reason)}"` : ""}>paused${
            l.pause_reason === "lid closed" ? " \u00b7 lid closed" : ""}</span>` : ""}${
          l.recovered ? '<span class="badge warn">recovered</span>' : ""}${
          l.state === "recording" ? '<span class="badge work">recording</span>' : ""}</div>
        <div class="t2">${esc(l.course ? l.course + " \u00b7 " : "")}${
          l.created_utc ? new Date(l.created_utc).toLocaleString() : ""}
          <button class="chip${l.pending_folder != null ? " pending" : ""}" data-move title="Move to another class">${
            esc(effFolder(l) || "Unfiled")}${l.pending_folder != null ? " (moving\u2026)" : ""} \u25be</button></div>
      </div>
      <div class="right">${l.duration_ms ? fmt(l.duration_ms / 1000) + "<br>" : ""}${
        l.words ? l.words.toLocaleString() + " words" : ""}</div>
      <button class="more" data-more title="More">\u22ef</button>`;
    li.onclick = (e) => {
      if (e.target.closest("[data-move],[data-more],.card-menu,.move-menu")) return;
      openLecture(l.id);
    };
    li.querySelector("[data-move]").onclick = (e) => {
      e.stopPropagation();
      openMoveMenu(li, l);
    };
    li.querySelector("[data-more]").onclick = (e) => {
      e.stopPropagation();
      openCardMenu(li, l);
    };
    ul.appendChild(li);
  }
}

function openCardMenu(li, l) {
  document.querySelectorAll(".card-menu,.move-menu").forEach((m) => m.remove());
  const menu = document.createElement("div");
  menu.className = "card-menu";
  menu.innerHTML = `
    <button data-act="open">Open</button>
    <button data-act="move">Move to class\u2026</button>
    <button data-act="rename">Rename\u2026</button>
    <button data-act="delete" class="danger-item">Delete</button>`;
  menu.querySelector('[data-act="open"]').onclick = () => openLecture(l.id);
  menu.querySelector('[data-act="move"]').onclick = (e) => {
    e.stopPropagation(); menu.remove(); openMoveMenu(li, l);
  };
  menu.querySelector('[data-act="rename"]').onclick = async () => {
    menu.remove();                 // an open menu holds back library re-renders
    const name = prompt("Rename lecture", l.title);
    if (!name || !name.trim() || name.trim() === l.title) return;
    try {
      await api(`/api/lecture/${encodeURIComponent(l.id)}`, {
        method: "PATCH", headers: { "content-type": "application/json" },
        body: JSON.stringify({ title: name.trim() }),
      });
      toast("Renamed."); loadLibrary();
    } catch (e) { toast(e.message || "Could not rename."); }
  };
  menu.querySelector('[data-act="delete"]').onclick = () => { menu.remove(); deleteLecture(l.id, l.title); };
  li.appendChild(menu);
  const close = (e) => {
    if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener("click", close); }
  };
  setTimeout(() => document.addEventListener("click", close), 0);
}
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

$("backBtn").onclick = showLibrary;

/* ---------------------------------------------------------- recording */
let folderTouched = false;
let folderDefault = "";           // what the dialog opened with, before any guess

function syncFolderUi() {
  const v = $("fFolder").value;
  const isNew = v === NEW_CLASS;
  $("fNewFolderWrap").classList.toggle("hidden", !isNew);
  $("fNewFolder").required = isNew;
  const name = isNew ? cleanFolder($("fNewFolder").value) : v;
  $("fFolderHint").textContent = isNew
    ? (name ? `Creates the class "${name}" and files this lecture there.` : "Type a name for the new class.")
    : (v ? `Saved in ~/Lectures/${v}.` : "Not filed under a class — you can move it later.");
}

$("fFolder").addEventListener("change", () => {
  folderTouched = true;
  syncFolderUi();
  if ($("fFolder").value === NEW_CLASS) $("fNewFolder").focus();
});
$("fNewFolder").addEventListener("input", syncFolderUi);
$("fTitle").addEventListener("input", () => {
  if (folderTouched) return;
  // Re-guess from the whole title each time, falling back to the default, so
  // a match made mid-word ("bi" -> BIO) does not stick once the title moves on.
  const want = guessFolder($("fTitle").value) ?? folderDefault;
  if ($("fFolder").value !== want) { $("fFolder").value = want; syncFolderUi(); }
});

/* ------------------------------------------------- first-run notice */
// The recording-law notice is shown once (per notice version) before the
// first recording or import. "Not now" closes it; Record/Import ask again.
let noticeOk = false;
function showNotice() {
  return new Promise((resolve) => {
    const dlg = $("noticeDlg");
    const done = (ok) => { dlg.close(); resolve(ok); };
    $("noticeOk").onclick = async () => {
      try { await jpost("/api/notice"); noticeOk = true; } catch (e) { toast(e.message); }
      done(noticeOk);
    };
    $("noticeLater").onclick = () => done(false);
    dlg.addEventListener("cancel", () => resolve(false), { once: true });
    if (!dlg.open) dlg.showModal();
  });
}
async function requireNotice() {
  if (noticeOk) return true;
  try { noticeOk = (await api("/api/notice")).accepted; } catch { /* ask anyway */ }
  return noticeOk || showNotice();
}

// Tiny renderer for our own Markdown legal files: headings, paragraphs,
// lists, **bold**, `code` and [links](url). Builds DOM nodes (no innerHTML);
// tables and anything else stay as preformatted text.
function inlineMd(text) {
  const frag = document.createDocumentFragment();
  const re = /\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)\s]+)\)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    frag.append(text.slice(last, m.index));
    if (m[1] != null) frag.append(el("b", "", m[1]));
    else if (m[2] != null) frag.append(el("code", "", m[2]));
    else if (/^https?:\/\//.test(m[4])) {
      const a = el("a", "", m[3]); a.href = m[4]; a.target = "_blank"; a.rel = "noopener";
      frag.append(a);
    } else frag.append(m[3]);
    last = re.lastIndex;
  }
  frag.append(text.slice(last));
  return frag;
}
function renderMd(md) {
  const out = document.createDocumentFragment();
  for (const block of md.replace(/\r/g, "").split(/\n\s*\n/)) {
    const lines = block.split("\n").filter((l) => l.trim());
    if (!lines.length) continue;
    const h = /^(#{1,3})\s+(.*)$/.exec(lines[0]);
    if (h && lines.length === 1) { out.append(el(h[1].length === 1 ? "h3" : "h4", "", h[2])); continue; }
    if (lines[0].startsWith("|")) {
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const table = el("table");
      lines.filter((l) => !/^\|?\s*:?-{3,}/.test(l)).forEach((l, i) => {
        const tr = el("tr");
        for (const c of cells(l)) { const td = el(i === 0 ? "th" : "td"); td.append(inlineMd(c)); tr.append(td); }
        table.append(tr);
      });
      out.append(table);
      continue;
    }
    if (/^\s*-\s/.test(lines[0])) {
      const ul = el("ul");
      let li = null;
      for (const l of lines) {
        if (/^\s*-\s/.test(l)) { li = el("li"); ul.append(li); li.append(inlineMd(l.replace(/^\s*-\s+/, ""))); }
        else if (li) li.append(" ", inlineMd(l.trim()));
      }
      out.append(ul);
      continue;
    }
    const p = el("p");
    p.append(inlineMd(lines.map((l) => l.trim()).join(" ")));
    out.append(p);
  }
  return out;
}

async function showLegal(doc) {
  const titles = { terms: "Terms & recording notice", privacy: "Privacy",
                   notices: "Third-party licences", license: "Lectern licence",
                   texts: "Full licence texts" };
  try {
    const r = await fetch(`/api/legal/${doc}`);
    if (!r.ok) throw new Error();
    const text = await r.text();
    $("legalText").replaceChildren(doc === "license" || doc === "texts" ? el("pre", "", text) : renderMd(text));
  } catch { $("legalText").textContent = "Could not load this document."; }
  $("legalTitle").textContent = titles[doc] || "Legal";
  if (!$("legalDlg").open) $("legalDlg").showModal();
}
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-legal]");
  if (!a) return;
  e.preventDefault();
  showLegal(a.dataset.legal);
});

$("newBtn").onclick = async () => {
  if (!(await requireNotice())) return;
  $("fTitle").value = ""; $("fNewFolder").value = "";
  folderTouched = false;
  const ms = $("micStatus");
  ms.className = "mic"; ms.textContent = "Checking microphone…";
  $("goBtn").disabled = true;
  await refreshFolders();
  // Default: the class open in the sidebar, else the class used last time.
  let def = L.sel === ALL ? "" : L.sel;
  if (def && !folderNames().includes(def)) def = "";   // renamed/deleted elsewhere
  if (!def) {
    let last = "";
    try { last = localStorage.getItem("lectern.lastFolder") || ""; } catch { /* private mode */ }
    if (folderNames().includes(last)) def = last;
  }
  folderDefault = def;
  $("fFolder").innerHTML = folderOptions(def, { withNew: true });
  $("fFolder").value = def;
  syncFolderUi();
  $("newDlg").showModal();
  $("fTitle").focus();            // Class comes first in the form; typing starts at the title
  try {
    const p = await jpost("/api/preflight");
    if (p.ok) {
      ms.className = "mic ok";
      ms.textContent = `Microphone ready — ${p.device} (level ${p.rms}).`;
      $("goBtn").disabled = false;
    } else {
      ms.className = "mic bad";
      ms.textContent = `${p.error} ${p.detail || ""}`.trim();
    }
  } catch (e) {
    ms.className = "mic bad"; ms.textContent = `Microphone check failed: ${e.message}`;
  }
};

$("newForm").addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value !== "go") return;
  const title = $("fTitle").value.trim();
  if (!title) return;
  let folder = $("fFolder").value;
  if (folder === NEW_CLASS) folder = cleanFolder($("fNewFolder").value);
  try {
    await jpost("/api/record/start", { title, course: "", folder });
    try { localStorage.setItem("lectern.lastFolder", folder); } catch { /* private mode */ }
    toast("Recording. You can close this tab — it keeps going.", 5200);
    poll();
  } catch (err) {
    toast(err.message || "Could not start recording.", 6000);
  }
});

$("stopBtn").onclick = async () => {
  $("stopBtn").disabled = true;
  try {
    const r = await jpost("/api/record/stop");
    toast("Recording saved. Transcribing now…", 4200);
    await loadLibrary();
    openLecture(r.lecture_id);
  } catch (e) { toast(e.message || "Stop failed."); }
  finally { $("stopBtn").disabled = false; }
};

$("pauseBtn").onclick = async () => {
  $("pauseBtn").disabled = true;
  try {
    await jpost("/api/record/pause");
    toast("Paused. Nothing is lost — resume any time from this lecture's page.", 5000);
    await loadLibrary();
    if (R.id === recLectureId) refreshLecture(false);
  } catch (e) { toast(e.message || "Could not pause."); }
  finally { $("pauseBtn").disabled = false; }
};

$("resumeBtn").onclick = async () => {
  const btn = $("resumeBtn");
  btn.disabled = true; btn.textContent = "starting…";
  try {
    await jpost("/api/record/resume", { lecture_id: R.id });
    toast("Recording again. You can browse elsewhere — it keeps going.", 5000);
    poll();
    refreshLecture(false);
  } catch (e) {
    toast(e.message || "Could not resume.", 6000);
  } finally {
    btn.disabled = false;
  }
};

$("finishBtn").onclick = async () => {
  const btn = $("finishBtn");
  btn.disabled = true;
  try {
    await jpost(`/api/lecture/${encodeURIComponent(R.id)}/process`);
    toast("Transcribing what was recorded…", 4200);
    refreshLecture(false);
  } catch (e) {
    toast(e.message || "Could not start transcribing.", 6000);
  } finally {
    btn.disabled = false;
  }
};

/* -------------------------------------------------------------- rename */
$("renameBtn").onclick = () => {
  const h = $("lecTitle"), inp = $("lecTitleEdit");
  inp.value = h.textContent;
  renameCommitted = false;
  h.classList.add("hidden"); inp.classList.remove("hidden");
  inp.focus(); inp.select();
};

let renameCommitted = false;

async function commitRename() {
  if (renameCommitted) return;         // Enter and the subsequent blur both fire
  renameCommitted = true;
  const h = $("lecTitle"), inp = $("lecTitleEdit");
  const title = inp.value.trim();
  inp.classList.add("hidden"); h.classList.remove("hidden");
  if (!title || title === h.textContent) return;
  const prev = h.textContent;
  h.textContent = title;
  try {
    await api(`/api/lecture/${encodeURIComponent(R.id)}`, {
      method: "PATCH", headers: { "content-type": "application/json" },
      body: JSON.stringify({ title }),
    });
    toast("Renamed.");
    loadLibrary();
  } catch (e) {
    h.textContent = prev;
    toast(e.message || "Could not rename.");
  }
}

$("lecTitleEdit").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); commitRename(); }
  else if (e.key === "Escape") {
    e.preventDefault();
    renameCommitted = true;            // suppress the blur that Escape causes
    $("lecTitleEdit").classList.add("hidden");
    $("lecTitle").classList.remove("hidden");
  }
});
$("lecTitleEdit").addEventListener("focus", () => { renameCommitted = false; });
$("lecTitleEdit").addEventListener("blur", commitRename);

/* -------------------------------------------------------------- delete */
async function deleteLecture(id, titleForConfirm) {
  $("deleteWhat").textContent = titleForConfirm;
  $("deleteDlg").dataset.id = id;
  $("deleteDlg").showModal();
}
$("deleteForm").addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value !== "go") return;
  const id = $("deleteDlg").dataset.id;
  try {
    await api(`/api/lecture/${encodeURIComponent(id)}`, { method: "DELETE" });
    toast("Deleted — recoverable from ~/Lectures/.trash for 7 days.");
    if (R.id === id) showLibrary(); else loadLibrary();
  } catch (err) {
    toast(err.message || "Could not delete that lecture.", 6000);
  }
});
$("deleteBtn").onclick = () => deleteLecture(R.id, R.lastLecture ? R.lastLecture.title : R.id);

/* ------------------------------------------------------------ downloads */
const DL_LABELS = { audio: "Audio (.m4a)", transcript: "Transcript only (.txt)", notes: "Notes only (.md)" };
$("dlBtn").onclick = (e) => {
  e.stopPropagation();
  const list = $("dlList");
  if (!list.classList.contains("hidden")) { list.classList.add("hidden"); return; }
  const d = R.lastLecture || {};
  list.textContent = "";
  const base = `/api/lecture/${encodeURIComponent(R.id)}/download/`;
  const add = (kind, title, sub, cls) => {
    const a = document.createElement("a");
    a.href = base + kind;
    if (cls) a.className = cls;
    const b = document.createElement("b"); b.textContent = title;
    a.appendChild(b);
    if (sub) { const s = document.createElement("span"); s.textContent = sub; a.appendChild(s); }
    list.appendChild(a);
  };
  const hasText = d.has_notes || d.has_transcript;
  if (!d.has_audio && !hasText) {
    list.innerHTML = `<div class="dl-none">Nothing ready to download yet.</div>`;
  } else {
    if (d.has_audio) add("audio", "Recording", "The audio file");
    if (hasText) add("markdown", "Markdown (.md)", "Notes + transcript in one file, for any notes app");
    const extra = [];
    if (d.has_transcript) extra.push(["transcript", DL_LABELS.transcript]);
    if (d.has_notes) extra.push(["notes", DL_LABELS.notes]);
    if (extra.length) {
      const h = document.createElement("div"); h.className = "dl-more"; h.textContent = "Individual files";
      list.appendChild(h);
      for (const [k, t] of extra) add(k, t, "", "small");
    }
  }
  list.classList.remove("hidden");
};
document.addEventListener("click", (e) => {
  if (!e.target.closest(".dl-menu")) $("dlList").classList.add("hidden");
});

let recWasActive = false, recActive = false, recLectureId = null;
let lastStopSeen = null;
let bannerShown = false, bannerRec = null;   // first-run "download a model" banner
async function poll() {
  let s;
  try { s = await api("/api/status"); } catch { return setTimeout(poll, 3000); }
  if (s.needs_model !== bannerShown) {
    bannerShown = s.needs_model;
    $("modelBanner").classList.toggle("hidden", !s.needs_model);
    if (s.needs_model) fillModelBanner();
  }
  const pill = $("recPill");
  recActive = !!s.recording;
  recLectureId = s.lecture_id || null;
  if (s.recording) {
    recWasActive = true;
    pill.classList.remove("hidden");
    $("newBtn").classList.add("hidden");
    $("recTime").textContent = clock(s.elapsed_s);
    $("meterFill").style.width = `${Math.round(Math.min(1, (s.level ?? 0) * 1.6) * 100)}%`;
    if (s.stalled) toast("The microphone stopped sending audio.", 5000);
    setTimeout(poll, 500);
  } else {
    pill.classList.add("hidden");
    $("newBtn").classList.remove("hidden");
    const ls = s.last_stop;
    if (ls && ls.paused && ls.at !== lastStopSeen && recWasActive) {
      const why = ls.reason === "lid closed" ? "the laptop lid was closed"
                : ls.reason === "system sleep" ? "the Mac went to sleep"
                : ls.reason === "recorder stopped unexpectedly" ? "the recorder stopped unexpectedly"
                : ls.reason === "disk full" ? "the disk is almost full"
                : null;
      if (why) toast(`Recording paused because ${why}. The mic is off — open the lecture to continue or finish it.`, 9000);
    }
    if (ls) lastStopSeen = ls.at;
    if (recWasActive) { recWasActive = false; loadLibrary(); if (R.id) refreshLecture(false); }
    setTimeout(poll, 2500);
  }
}

/* ------------------------------------------------------------- start */
const OPEN_SETTINGS = location.hash === "#settings";   // the menu bar's "Change shortcut…"
requireNotice();                         // first launch: show the notice right away
(function start() {
  const m = location.hash.match(/^#\/l\/(.+)$/);
  if (m) openLecture(decodeURIComponent(m[1])); else showLibrary();
  poll();
})();

/* ---------------------------------------------------------- settings */
let dlTimer = 0;
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
const mbText = (mb) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`);

async function loadSettings() {
  let s;
  try { s = await api("/api/system"); }
  catch { $("setHw").textContent = "Could not read this computer's details."; return; }
  const hw = s.hardware;
  const accel = { metal: "Apple GPU", cuda: "NVIDIA GPU", cpu: "no GPU" }[hw.accelerator] || hw.accelerator;
  $("setHw").textContent = `This computer: ${s.device_label}, ${hw.cpu_cores} cores, ${accel}.`;
  $("setRec").textContent = `Recommended: ${s.recommended_label}. ${s.recommended_why}`;

  const ul = $("modelList"); ul.replaceChildren();
  const busy = !!(s.download && s.download.running);
  for (const m of s.models) {
    const li = el("li", m.id === s.active.id ? "sel" : "");
    const main = el("div", "m-main");
    const name = el("div", "m-name", m.label);
    if (m.recommended) name.append(el("span", "tag", "recommended"));
    if (m.id === s.active.id) name.append(el("span", "tag", "in use"));
    if (!m.fits) name.append(el("span", "tag", "may be too heavy here"));
    const eta = m.min_per_hour ? ` About ${m.min_per_hour} min per hour of lecture here.` : "";
    main.append(name, el("div", "m-note", `${m.note} ${mbText(m.size_mb)} download.${eta}`));
    li.append(main);
    const btn = el("button", m.installed ? "" : "primary");
    if (m.installed) {
      btn.textContent = m.id === s.active.id ? "Selected" : "Use this";
      btn.disabled = m.id === s.active.id;
      btn.onclick = async () => {
        try { await jpost("/api/models/select", { id: m.id }); toast("Speech model changed."); }
        catch (e) { toast(e.message); }
        loadSettings();
      };
    } else {
      btn.textContent = `Download ${mbText(m.size_mb)}`;
      btn.disabled = busy;
      btn.onclick = async () => {
        try { await jpost("/api/models/download", { id: m.id }); } catch (e) { toast(e.message); }
        loadSettings();
      };
    }
    li.append(btn);
    ul.append(li);
  }

  const d = s.download || {};
  $("dlProgress").classList.toggle("hidden", !busy);
  $("dlError").classList.toggle("hidden", !d.error);
  $("dlError").textContent = d.error || "";
  if (busy) {
    const pct = d.total ? Math.min(100, Math.round(100 * d.bytes / d.total)) : 0;
    $("dlLabel").textContent = `Downloading… ${pct}% (${mbText(Math.round(d.bytes / 1e6))} of ${mbText(Math.round(d.total / 1e6))})`;
    $("dlFill").style.width = pct + "%";
  }
  clearTimeout(dlTimer);
  if (busy && $("settingsDlg").open) dlTimer = setTimeout(loadSettings, 800);
}

$("settingsBtn").onclick = () => { $("settingsDlg").showModal(); loadSettings(); loadShortcut(); loadStorage(); };

/* ------------------------------------------------------------ storage */

async function loadStorage() {
  const box = $("storage");
  let r;
  try { r = await api("/api/storage"); }
  catch { box.replaceChildren(el("p", "hint", "Could not measure storage.")); return; }
  const disk = r.disk, pct = (b) => (100 * b / disk.total);
  const head = el("div", "st-head");
  head.append(el("b", "", `${bytesText(r.total_bytes)}`),
              el("span", "", `used by Lectern \u00b7 ${bytesText(disk.free)} free of ${bytesText(disk.total)}`));
  // Whole-disk bar: Lectern's parts in colour, everything else grey, free empty.
  const bar = el("div", "st-bar");
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", `Lectern uses ${bytesText(r.total_bytes)}; ` +
    `${bytesText(disk.free)} of ${bytesText(disk.total)} free on this disk`);
  for (const p of r.parts) {
    if (!p.bytes) continue;
    const seg = el("i", pct(p.bytes) < 0.4 ? "tiny" : "");
    seg.style.width = `${pct(p.bytes)}%`;
    seg.style.background = `var(--st-${p.key})`;
    seg.title = `${p.label}: ${bytesText(p.bytes)}`;
    bar.append(seg);
  }
  const other = Math.max(0, disk.used - r.total_bytes);
  const os_ = el("i"); os_.style.width = `${pct(other)}%`; os_.style.background = "var(--st-other)";
  os_.title = `Other files and apps: ${bytesText(other)}`;
  bar.append(os_);
  const scale = el("div", "st-scale");
  scale.append(el("span", "", `Other apps and files: ${bytesText(other)}`), el("span", "", `${bytesText(disk.free)} free`));
  const ul = el("ul", "st-list");
  for (const p of r.parts) {
    const li = el("li");
    const dot = el("span", "st-dot"); dot.style.background = `var(--st-${p.key})`;
    const name = el("span", "st-name", p.label);
    name.append(el("small", "", p.detail));
    li.append(dot, name, el("span", "st-size", bytesText(p.bytes)));
    ul.append(li);
  }
  box.replaceChildren(head, bar, scale, ul);
  if (r.shared_bytes) {
    box.append(el("p", "st-note",
      `Also using ${r.shared_detail} from another app's folder (not counted above).`));
  }
}
if (OPEN_SETTINGS) setTimeout(() => $("settingsBtn").click(), 0);

/* ------------------------------------------------- quick-record shortcut */
const SC_GLYPH = { ctrl: "\u2303", opt: "\u2325", shift: "\u21e7", cmd: "\u2318" };
const scLabel = (sc) => ["ctrl", "opt", "shift", "cmd"].filter((m) => sc.mods.includes(m))
  .map((m) => SC_GLYPH[m]).join("") + sc.key;
let shortcut = null, scListening = false;

function scHint() {
  const conflict = shortcut.enabled && shortcut.key === "L" && shortcut.mods.join() === "cmd";
  $("scHint").textContent = scListening
    ? "Press the new combination (hold \u2318, \u2303 or \u2325 plus a letter or digit). Esc cancels."
    : conflict
      ? "Note: \u2318L normally jumps to a browser's address bar. While the helper runs, it records instead."
      : "";
}

async function loadShortcut() {
  try { shortcut = await api("/api/shortcut"); } catch { return; }
  $("scEnabled").checked = shortcut.enabled;
  $("scBtn").textContent = scLabel(shortcut);
  scHint();
}

async function saveShortcut(next) {
  try {
    shortcut = await jpost("/api/shortcut", next);
    toast(`Quick record: ${shortcut.enabled ? scLabel(shortcut) : "off"}`);
  } catch (e) { toast(e.message || "Could not save the shortcut."); }
  loadShortcut();
}

$("scEnabled").onchange = () => saveShortcut({ ...shortcut, enabled: $("scEnabled").checked });
$("scBtn").onclick = () => {
  scListening = true;
  $("scBtn").classList.add("listening");
  $("scBtn").textContent = "Press keys\u2026";
  scHint();
};
document.addEventListener("keydown", (e) => {
  if (!scListening) return;
  e.preventDefault();
  e.stopPropagation();
  const stop = () => { scListening = false; $("scBtn").classList.remove("listening"); };
  if (e.key === "Escape") { stop(); return loadShortcut(); }
  // e.code is layout-independent ("KeyL", "Digit5"), matching the helper's key codes.
  const m = /^(?:Key([A-Z])|Digit([0-9]))$/.exec(e.code);
  const mods = [e.metaKey && "cmd", e.ctrlKey && "ctrl", e.altKey && "opt", e.shiftKey && "shift"].filter(Boolean);
  if (!m) return;                                   // still holding modifiers
  if (!mods.some((x) => x !== "shift")) { toast("Hold \u2318, \u2303 or \u2325 too."); return; }
  stop();
  saveShortcut({ enabled: true, key: m[1] || m[2], mods });
}, true);
async function fillModelBanner() {
  try {
    const sys = await api("/api/system");
    bannerRec = sys.recommended;
    const dev = sys.hardware.device || {}, ram = sys.hardware.ram_total_mb;
    const specs = [dev.chip, ram && `${Math.round(ram / 1024)} GB`].filter(Boolean).join(", ");
    const what = dev.name ? `this ${dev.name}${specs ? ` (${specs})` : ""}` : "this computer";
    const label = sys.recommended_label.replace(/\s*\(default\)$/, "");
    $("modelBannerText").textContent =
      `Before your first lecture, download a speech model. Recommended for ${what}: ` +
      `${label}, ${mbText(sys.recommended_size_mb)}.`;
    $("modelBannerGo").textContent = `Download ${label}`;
    $("modelBannerGo").classList.remove("hidden");
  } catch { /* keep the generic text */ }
}
$("modelBannerGo").onclick = async () => {
  if (!bannerRec) return;
  try { await jpost("/api/models/download", { id: bannerRec }); } catch (e) { return toast(e.message); }
  $("settingsBtn").click();                       // shows the download progress
};
$("modelBannerBtn").onclick = () => $("settingsBtn").click();
$("settingsDlg").addEventListener("close", () => clearTimeout(dlTimer));
$("dlCancel").onclick = async () => { await jpost("/api/models/download/cancel"); setTimeout(loadSettings, 500); };

// Plain link-outs; they open in the browser and send nothing by themselves.
// An empty URL hides its link (e.g. donations until the page exists).
const REPO = "https://github.com/lqoro/Lectern";
const HELP_LINKS = [
  ["Report a bug", `${REPO}/issues/new?template=bug.yml`],
  ["Request a feature", `${REPO}/issues/new?template=feature.yml`],
  ["Source code", REPO],
  ["Donate", ""],
];
for (const [label, href] of HELP_LINKS) {
  if (!href) continue;
  const a = el("a", "", label);
  a.href = href; a.target = "_blank"; a.rel = "noopener";
  $("helpLinks").append(a);
}

async function diagnosticsText() {
  const d = await api("/api/diagnostics");
  const hw = d.hardware;
  return [
    `${d.app}`,
    `Device: ${[hw.device?.name, hw.device?.chip].filter(Boolean).join(", ") || "unknown"}`,
    `OS: ${hw.os} ${hw.machine}, Python ${hw.python}`,
    `Memory: ${hw.ram_total_mb} MB total, ${hw.ram_available_mb} MB available`,
    `CPU: ${hw.cpu_cores} cores / ${hw.cpu_threads} threads, accelerator ${hw.accelerator}`,
    `Speech model: ${d.speech_model.file || "none"} (${d.speech_model.why}), ${hw.whisper_threads} threads`,
    `whisper-cli: ${d.whisper_cli}`, `ffmpeg: ${d.ffmpeg}`,
    `Browser: ${navigator.userAgent}`,
  ].join("\n");
}

// Safari only lets a page write the clipboard during the click itself, not
// after an await, so the write starts synchronously with a promised blob.
function copyPromisedText(textPromise) {
  if (!navigator.clipboard || typeof ClipboardItem === "undefined") return Promise.reject(new Error("no clipboard"));
  const blob = textPromise.then((t) => new Blob([t], { type: "text/plain" }));
  return navigator.clipboard.write([new ClipboardItem({ "text/plain": blob })]);
}

// Fallback copy. The textarea goes INSIDE the open dialog: everything outside
// a modal dialog is inert, so a textarea on <body> can't be selected and
// execCommand silently copies nothing.
function copyViaTextarea(text, host) {
  const ta = el("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.cssText = "position:fixed;opacity:0;pointer-events:none";
  host.append(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch { ok = false; }
  ta.remove();
  return ok;
}

$("diagBtn").onclick = async () => {
  const textP = diagnosticsText();
  // Started before any await, while the click still counts as a user gesture.
  const firstTry = copyPromisedText(textP).then(() => true, () => false);
  const host = $("settingsDlg");
  let text;
  try { text = await textP; }
  catch (e) { return toast(e.message || "Could not collect diagnostics."); }
  let copied = await firstTry;
  if (!copied) {
    try { await navigator.clipboard.writeText(text); copied = true; } catch { /* next */ }
  }
  if (!copied) copied = copyViaTextarea(text, host);
  const box = $("diagText");
  if (copied) {
    box.classList.add("hidden");
    toast("Diagnostics copied. Paste them into your bug report.");
  } else {
    // Never claim a copy that didn't happen: show it to copy by hand.
    box.value = text;
    box.classList.remove("hidden");
    box.focus();
    box.select();
    toast("Your browser blocked copying. The details are selected below: press \u2318C.", 6000);
  }
};

/* ------------------------------------------------------------ import */
// The file goes up as the raw request body (not multipart) so the backend can
// stream it straight to disk. XMLHttpRequest, because fetch has no upload progress.
function uploadOne(file, title, folder, onPct) {
  return new Promise((resolve, reject) => {
    const q = new URLSearchParams({ title, folder, filename: file.name });
    const x = new XMLHttpRequest();
    x.open("POST", `/api/import?${q}`);
    x.setRequestHeader("content-type", "application/octet-stream");
    x.upload.onprogress = (e) => { if (e.lengthComputable) onPct(Math.round(100 * e.loaded / e.total)); };
    x.onload = () => {
      let body = {};
      try { body = JSON.parse(x.responseText); } catch { /* not json */ }
      if (x.status >= 200 && x.status < 300) resolve(body);
      else reject(new Error(body.error || `upload failed (${x.status})`));
    };
    x.onerror = () => reject(new Error("upload failed — is Lectern still running?"));
    x.send(file);
  });
}

const titleFromName = (n) => n.replace(/\.[^.]+$/, "").replace(/[_]+/g, " ").trim() || "Imported lecture";

$("importBtn").onclick = async () => {
  if (!(await requireNotice())) return;
  $("iFiles").value = ""; $("iTitle").value = "";
  $("iProgress").classList.add("hidden");
  $("iGo").disabled = false;
  await refreshFolders();
  const def = L.sel === ALL || !folderNames().includes(L.sel) ? "" : L.sel;
  $("iFolder").innerHTML = folderOptions(def, { withNew: true });
  $("iFolder").value = def;
  $("iNewFolder").value = "";
  syncImportFolderUi();
  $("iTitleWrap").classList.remove("hidden");
  $("importDlg").showModal();
};

function syncImportFolderUi() {
  const v = $("iFolder").value;
  const isNew = v === NEW_CLASS;
  $("iNewFolderWrap").classList.toggle("hidden", !isNew);
  $("iNewFolder").required = isNew;
  const name = isNew ? cleanFolder($("iNewFolder").value) : v;
  $("iFolderHint").textContent = isNew
    ? (name ? `Creates the class "${name}" and files the import there.` : "Type a name for the new class.")
    : (v ? `Saved in ~/Lectures/${v}.` : "Not filed under a class — you can move it later.");
}

$("iFolder").addEventListener("change", () => {
  syncImportFolderUi();
  if ($("iFolder").value === NEW_CLASS) $("iNewFolder").focus();
});
$("iNewFolder").addEventListener("input", syncImportFolderUi);

$("iFiles").addEventListener("change", () => {
  const fs = $("iFiles").files;
  // One title only makes sense for one file; several are titled by file name.
  $("iTitleWrap").classList.toggle("hidden", fs.length > 1);
  if (fs.length === 1 && !$("iTitle").value) $("iTitle").placeholder = titleFromName(fs[0].name);
});

$("importForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = [...$("iFiles").files];
  if (!files.length) return;
  let folder = $("iFolder").value;
  if (folder === NEW_CLASS) {
    folder = cleanFolder($("iNewFolder").value);
    if (!folder) { toast("Type a name for the new class."); $("iNewFolder").focus(); return; }
  }
  $("iGo").disabled = true;
  $("iProgress").classList.remove("hidden");
  let first = null, failed = 0;
  for (const [i, f] of files.entries()) {
    const title = files.length === 1 && $("iTitle").value.trim() ? $("iTitle").value.trim() : titleFromName(f.name);
    $("iLabel").textContent = files.length > 1 ? `Uploading ${i + 1} of ${files.length}: ${f.name}` : `Uploading ${f.name}`;
    try {
      const r = await uploadOne(f, title, folder, (p) => {
        $("iPct").textContent = `${p}%`; $("iFill").style.width = `${p}%`;
      });
      first = first || r.lecture_id;
    } catch (err) { failed++; toast(`${f.name}: ${err.message}`, 7000); }
  }
  $("importDlg").close();
  loadLibrary();
  if (first) {
    toast(failed ? `Imported ${files.length - failed} of ${files.length}. Transcribing now…`
                 : "Imported. Transcribing now — you can leave this page.", 5200);
    openLecture(first);
  }
});
