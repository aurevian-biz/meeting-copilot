#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 meeting-copilot contributors

/* dom_render_test.js — verify incremental rendering of the caption page with a minimal DOM mock.
 *
 * Why not eyeball it: the bug shows as "a new sentence arrives and the whole previous line vanishes";
 * it reproduces only with a long sentence plus timing, and by eye you cannot tell a data problem from
 * a layout shift. This runs the page's real <script> and asserts on DOM node identity and content.
 *
 * Run:
 *   node tools/dom_render_test.js
 *
 * The page script is extracted from caption_server.py (single source of truth), not a stale
 * tools/_page_check.js — that file is deliberately left out of the public repo.
 */
const fs = require("fs");
const path = require("path");

// ---------- minimal DOM ----------
class N {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this._text = "";
    this.className = "";
    this.parent = null;
    this.style = {};
  }
  get isConnected() { return !!this.parent; }
  appendChild(c) {
    if (c.parent) c.remove();
    c.parent = this;
    this.children.push(c);
    return c;
  }
  append(...cs) { for (const c of cs) this.appendChild(c); }
  insertBefore(c, ref) {
    if (!ref) return this.appendChild(c);
    if (c.parent) c.remove();
    const i = this.children.indexOf(ref);
    c.parent = this;
    this.children.splice(i < 0 ? this.children.length : i, 0, c);
    return c;
  }
  prepend(...cs) { for (const c of cs.slice().reverse()) this.insertBefore(c, this.firstElementChild); }
  // Fake layout as "child index x 100px": enough to verify the direction and size of scroll compensation
  getBoundingClientRect() {
    if (!this.parent) return { top: 0, height: 0 };
    return { top: this.parent.children.indexOf(this) * 100, height: 100 };
  }
  remove() {
    if (!this.parent) return;
    const p = this.parent;
    p.children = p.children.filter((x) => x !== this);
    this.parent = null;
  }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  // The page only uses this to insert a placeholder line ("等待声音…"); good enough
  set innerHTML(v) {
    this.children = []; this._text = "";
    const m = /class="([^"]+)"/.exec(String(v));
    if (m) { const c = new N("div"); c.className = m[1]; this.appendChild(c); }
  }
  get innerHTML() { return ""; }
  get firstElementChild() { return this.children[0] || null; }
  get lastElementChild() { return this.children[this.children.length - 1] || null; }
  get nextElementSibling() {
    if (!this.parent) return null;
    const i = this.parent.children.indexOf(this);
    return this.parent.children[i + 1] || null;
  }
  querySelector(sel) {
    const cls = sel.replace(/^\./, "");
    const walk = (n) => {
      for (const c of n.children) {
        if ((c.className || "").split(/\s+/).includes(cls)) return c;
        const r = walk(c);
        if (r) return r;
      }
      return null;
    };
    return walk(this);
  }
}

const main = new N("main");
main.className = "";

const doc = {
  createElement: (t) => new N(t),
  createTextNode: (t) => { const n = new N("#text"); n._text = String(t); return n; },
  querySelector: (sel) => {
    if (sel === "#main") return main;
    if (!doc._cache[sel]) doc._cache[sel] = new N("div");
    return doc._cache[sel];
  },
  body: new N("body"),
  _cache: {},
};
doc.body.scrollHeight = 1000;

const win = {
  scrollY: 0,
  innerHeight: 800,
  scrollTo: (x, y) => { win.scrollY = y || 0; },
  scrollBy: (x, y) => { win.scrollY += (y || 0); },
  EventSource: class { constructor() {} close() {} },
};

// ---------- run the page script + assertions ----------
// Extract <script> from caption_server.py's PAGE right now: change the page and you test the new behavior, never a stale copy
const pagePy = fs.readFileSync(path.join(__dirname, "..", "caption_server.py"), "utf8");
const hit = pagePy.match(/PAGE = r?"""[\s\S]*?<script>([\s\S]*?)<\/script>/);
if (!hit) throw new Error("caption_server.py 的 PAGE 里找不到 <script>");
const src = hit[1];
const test = `
  const M = document.querySelector('#main');
  const A = [];
  const ok = (cond, msg) => A.push([!!cond, msg]);

  function row(seq, extra) {
    return Object.assign({
      seq: seq, at: "10:0" + (seq % 10), dur: 2.4, lang: "ja",
      text: "原文" + seq, trans: "", pending: true, lag_ms: 300,
    }, extra || {});
  }
  const nodeOf = (seq) => dom.get(seq) && dom.get(seq).root;

  // 1) Lay down 35 rows: the window holds only the last 30, newest at the top
  for (let i = 1; i <= 35; i++) pushLines([row(i, { trans: "译文" + i, pending: false })]);
  render();
  ok(M.children.length === 30, "30 行窗口，实际 " + M.children.length);
  ok(M.firstElementChild === nodeOf(35), "最新的一条在最上（35）");
  ok(M.lastElementChild === nodeOf(6), "最老的一条沉到底部（6）");
  ok(!dom.has(5) && dom.has(6) && dom.has(35), "最早的 5 行被回收（6..35 保留）");

  // 2) The last row is a long sentence whose translation grows 20 times — earlier rows must stay untouched
  const before30 = nodeOf(30);
  const before34 = nodeOf(34);
  const text30 = before30 ? before30.textContent : "";
  for (let k = 1; k <= 20; k++) {
    applyUpd([{ seq: 35, trans: "很长的译文".repeat(k), final: false }]);
    render();
  }
  ok(nodeOf(30) === before30, "长句流式 20 次后，第 30 行的 DOM 节点还是同一个对象");
  ok(nodeOf(34) === before34, "倒数第二行节点未被重建");
  ok(nodeOf(30) && nodeOf(30).textContent === text30, "第 30 行文本一个字都没变");

  // 3) Next sentence arrives: earlier rows must not be deleted or re-created
  pushLines([row(36, { text: "下一句原文" })]);
  render();
  ok(nodeOf(30) === before30, "新行到来后，第 30 行仍未被重建");
  ok(dom.has(36) && dom.has(7) && !dom.has(6), "窗口随新行滑动：6 出、36 进");
  ok(M.firstElementChild === nodeOf(36) && M.children[1] === nodeOf(35),
     "新行插到最上，原来的 35 退到第二位");
  ok(nodeOf(35) && nodeOf(35).textContent.includes("很长的译文"), "上一句（35）译文完整保留");

  // 4) Previous sentence still streaming, next one already here: interleaved updates must not clobber each other
  const n35 = nodeOf(35);
  const n36 = nodeOf(36);
  applyUpd([{ seq: 35, trans: "很长的译文".repeat(20) + "，继续", final: false }]);
  applyUpd([{ seq: 36, trans: "下一句起头", final: false }]);
  render();
  ok(nodeOf(35) === n35 && nodeOf(36) === n36, "交错更新不换节点");
  ok(n35.textContent.includes("继续"), "长句后续增量写进去了");
  ok(n36.textContent.includes("下一句起头"), "新句译文也写进去了");

  // 5) pending / final state switch: must not wipe the translation already shown
  applyUpd([{ seq: 36, trans: "下一句起头", final: true, trans_ms: 800, ttft_ms: 300 }]);
  render();
  ok(n36.textContent.includes("下一句起头"), "final 后译文保留");
  const tr36 = dom.get(36).tr;
  ok(tr36.className === "tr", "final 后不再显示光标类名：" + tr36.className);

  // 6) Draft row: appears at the top (it is the newest) and does not disturb existing rows
  draft = "正在说的内容";
  render();
  ok(draftEl && draftEl.isConnected, "草稿行已插入");
  ok(M.firstElementChild === draftEl, "草稿行在最上");
  ok(M.children[1] === nodeOf(36), "正式行退到第二位");
  ok(nodeOf(30) === before30, "草稿行插入不影响已有行");
  draft = "";
  render();
  ok(!draftEl.isConnected, "草稿清空后节点移除");
  ok(M.firstElementChild === nodeOf(36), "草稿移除后 36 回到最上");

  // 7) Scrolled down reading old rows: a new row pushes content down, so scroll must be compensated by the same amount; the view must not jump
  window.scrollY = 300;
  const anchorNode = M.firstElementChild;
  const anchorPos = anchorNode.getBoundingClientRect().top;
  pushLines([row(37, { text: "又一句" })]);
  render();
  const moved = anchorNode.getBoundingClientRect().top - anchorPos;
  ok(moved > 0, "新行把上面的内容往下推了 " + moved + "px");
  ok(Math.abs(window.scrollY - (300 + moved)) < 1,
     "滚动补偿到位，scrollY=" + window.scrollY + "（期望 " + (300 + moved) + "）");
  window.scrollY = 0;
  render();
  ok(window.scrollY === 0, "贴顶时不做补偿，新行自然把内容往下推");

  // 8) Clear the screen
  lines = []; idx = new Map(); draft = "";
  dom.clear(); draftEl = null;
  M.innerHTML = "";
  render();
  ok(M.querySelector(".empty") !== null, "空列表回到「等待声音…」");

  return A;
`;
const results = new Function("document", "window", "EventSource", src + test)(doc, win, win.EventSource);

let bad = 0;
for (const [pass, msg] of results) {
  if (!pass) bad++;
  console.log((pass ? "[ok]   " : "[FAIL] ") + msg);
}
console.log(bad ? `\n${bad} 项失败` : "\nALL PASS");
process.exit(bad ? 1 : 0);
