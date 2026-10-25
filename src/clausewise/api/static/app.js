/* clausewise dashboard: vanilla JS, no build step. Talks to /api/* on the same origin. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const apiKey = new URLSearchParams(location.search).get("key");
  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers["X-API-Key"] = apiKey;
  let current = null;

  async function api(path, body) {
    const res = await fetch(path, body ? { method: "POST", headers, body: JSON.stringify(body) } : { headers });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* keep statusText */ }
      throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    }
    return res.json();
  }

  async function loadHealth() {
    try {
      const h = await api("/health");
      $("status").textContent = `v${h.version} | embeddings: ${h.embeddings}${h.auth_enabled ? " | auth on" : ""}`;
    } catch (e) { $("status").textContent = "health check failed"; }
  }

  async function loadSamples() {
    try {
      const list = await api("/api/samples");
      for (const s of list) {
        const o = document.createElement("option");
        o.value = s.contract_id; o.textContent = `${s.contract_id}  ${s.title}`;
        $("sample").appendChild(o);
      }
    } catch (e) { $("hint").textContent = `Samples unavailable: ${e.message}`; }
  }

  $("sample").addEventListener("change", async (ev) => {
    if (!ev.target.value) return;
    const s = await api(`/api/samples/${ev.target.value}`);
    $("text").value = s.text;
  });

  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".tabpanel").forEach((p) => { p.hidden = p.id !== `tab-${t.dataset.tab}`; });
  }));

  function renderRisks(r) {
    const el = $("tab-risks");
    if (!r.risks.length) { el.innerHTML = '<p class="empty">No playbook findings. Every risk rule was checked against every clause.</p>'; return; }
    const byClause = Object.fromEntries(r.redlines.map((x) => [`${x.clause_id}:${x.risk_type}`, x]));
    el.innerHTML = r.risks.map((f) => {
      const red = byClause[`${f.clause_id}:${f.risk_type}`];
      const diff = red ? red.diff.split("\n").map((l) => `<span class="${l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : ""}">${esc(l)}</span>`).join("\n") : "";
      return `<article class="card">
        <h3><span class="sev ${f.severity}">${f.severity}</span>${esc(f.risk_type.replace(/_/g, " "))} <span class="kv">in <b>${esc(f.clause_id)}</b> (weight ${f.weight})</span></h3>
        <div class="kv">${esc(f.explanation)}</div>
        <div class="evidence">${esc(f.evidence)}</div>
        <div class="kv">Playbook position: <b>${esc(f.playbook_position)}</b></div>
        ${red ? `<details><summary>Redline suggestion</summary><div class="kv">${esc(red.proposed)}</div><pre class="diff">${diff}</pre></details>` : ""}
      </article>`;
    }).join("");
  }

  function renderClauses(r) {
    const flagged = new Set(r.risks.map((f) => f.clause_id));
    $("tab-clauses").innerHTML = r.clauses.map((c) => `<div class="clause ${flagged.has(c.clause_id) ? "flagged" : ""}">
      <span class="id">${esc(c.clause_id)}</span><span class="label">${esc(c.label)}</span>
      <span class="snippet">${esc(c.heading ? c.heading + ": " : "")}${esc(c.text.replace(/\s+/g, " ").slice(0, 220))}</span>
      <span class="conf">${(c.confidence * 100).toFixed(0)}%</span></div>`).join("");
  }

  function renderEntities(r) {
    if (!r.entities.length) { $("tab-entities").innerHTML = '<p class="empty">No entities found.</p>'; return; }
    $("tab-entities").innerHTML = `<table><thead><tr><th>Type</th><th>Normalised value</th><th>Clause</th></tr></thead><tbody>${r.entities.map((e) => `<tr><td>${esc(e.type)}</td><td class="mono">${esc(e.value)}</td><td class="mono">${esc(e.clause_id || "")}</td></tr>`).join("")}</tbody></table>`;
  }

  const SUGGESTED = ["Which law governs this agreement?", "How much notice is required to terminate?", "Does the agreement renew automatically?", "What is the cap on liability?", "Who are the parties to this agreement?", "How long do the confidentiality obligations last?"];

  function renderSuggested() {
    $("suggested").innerHTML = SUGGESTED.map((q) => `<span class="chip">${esc(q)}</span>`).join("");
    $("suggested").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { $("question").value = c.textContent; ask(); }));
  }

  function renderAnswer(a) {
    const el = $("answer");
    if (a.abstained) {
      el.innerHTML = `<p class="empty"><span class="badge warn">abstained</span>No sentence in the document supports an answer, so none was invented. Retrieved: ${esc(a.retrieved.join(", "))}.</p>`;
      return;
    }
    el.innerHTML = (a.short_answer ? `<p><span class="badge">short answer</span><b>${esc(a.short_answer)}</b> <span class="kv">(${esc(a.answer_type)})</span></p>` : "") +
      a.sentences.map((s) => `<div class="answer-sentence">${esc(s.text)}<span class="cite">[${esc(s.citations.join(", "))}]</span> <span class="kv">grounded, support ${s.support.toFixed(2)}</span></div>`).join("") +
      (a.rejected.length ? `<p class="kv">${a.rejected.length} sentence(s) rejected by the grounding checker.</p>` : "") +
      `<p class="kv">Retrieved clauses: ${esc(a.retrieved.join(", "))}. Method: ${esc(a.method)}.</p>`;
  }

  async function analyze() {
    const text = $("text").value;
    if (text.trim().length < 20) { $("hint").textContent = "Paste a contract first (at least 20 characters)."; return; }
    $("analyze").disabled = true; $("hint").textContent = "Analysing...";
    try {
      const r = await api("/api/analyze", { text });
      current = r;
      $("results").hidden = false;
      $("score").textContent = r.risk_score;
      $("ring").style.setProperty("--pct", r.risk_score);
      $("ring").style.setProperty("--ring-color", r.risk_score >= 50 ? "var(--high)" : r.risk_score >= 20 ? "var(--medium)" : "var(--ok)");
      $("ctype").textContent = r.contract_type;
      $("nclauses").textContent = `${r.clauses.length} clauses`;
      $("nrisks").textContent = `${r.risks.length} finding${r.risks.length === 1 ? "" : "s"}`;
      $("summary").textContent = r.summary;
      renderRisks(r); renderClauses(r); renderEntities(r); renderSuggested();
      $("answer").innerHTML = "";
      $("hint").textContent = `Analysed ${r.clauses.length} clauses.`;
    } catch (e) { $("hint").innerHTML = `<span class="error">${esc(e.message)}</span>`; }
    finally { $("analyze").disabled = false; }
  }

  async function ask() {
    const question = $("question").value.trim();
    if (!current || question.length < 3) return;
    $("askbtn").disabled = true;
    try { renderAnswer(await api("/api/ask", { text: $("text").value, question, mode: $("mode").value })); }
    catch (e) { $("answer").innerHTML = `<p class="error">${esc(e.message)}</p>`; }
    finally { $("askbtn").disabled = false; }
  }

  $("analyze").addEventListener("click", analyze);
  $("askbtn").addEventListener("click", ask);
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
  loadHealth(); loadSamples();
})();
