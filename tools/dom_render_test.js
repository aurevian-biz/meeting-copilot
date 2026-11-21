#!/usr/bin/env node
/* dom_render_test.js —— 用极简 DOM mock 验证字幕页的增量渲染。
 *
 * 为什么不靠人眼：这个 bug 的表现是「下一句一来，上一句整行消失」，
 * 复现要等长句 + 时序，靠肉眼极难判断是数据问题还是渲染位移。
 * 这里把页面真实的 <script> 抽出来跑，直接断言 DOM 节点的身份和内容。
 *
 * 跑法：
 *   node tools/dom_render_test.js
 *
 * 页面脚本直接从 caption_server.py 抽（单一真相源），不再依赖一份可能过期的
 * tools/_page_check.js —— 那个文件在公开仓里是故意不放的。
 */
const fs = require("fs");
const path = require("path");

// ---------- 极简 DOM ----------
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
  // 用「第几个孩子 × 100px」模拟布局：足够验证滚动补偿的方向和数值
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
  // 页面只会用它来塞一行占位（"等待声音…"），够用就行
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

// ---------- 跑页面脚本 + 断言 ----------
// 从 caption_server.py 的 PAGE 里现抽 <script>：改了页面就有新行为，不会测到旧副本
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

  // 1) 铺 35 行：窗口只放最后 30 行，且新的在最上
  for (let i = 1; i <= 35; i++) pushLines([row(i, { trans: "译文" + i, pending: false })]);
  render();
  ok(M.children.length === 30, "30 行窗口，实际 " + M.children.length);
  ok(M.firstElementChild === nodeOf(35), "最新的一条在最上（35）");
  ok(M.lastElementChild === nodeOf(6), "最老的一条沉到底部（6）");
  ok(!dom.has(5) && dom.has(6) && dom.has(35), "最早的 5 行被回收（6..35 保留）");

  // 2) 最后一行是长句，译文持续增长 20 次 —— 前面行的节点必须原封不动
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

  // 3) 下一句到来：前面的行不能被删，也不能换对象
  pushLines([row(36, { text: "下一句原文" })]);
  render();
  ok(nodeOf(30) === before30, "新行到来后，第 30 行仍未被重建");
  ok(dom.has(36) && dom.has(7) && !dom.has(6), "窗口随新行滑动：6 出、36 进");
  ok(M.firstElementChild === nodeOf(36) && M.children[1] === nodeOf(35),
     "新行插到最上，原来的 35 退到第二位");
  ok(nodeOf(35) && nodeOf(35).textContent.includes("很长的译文"), "上一句（35）译文完整保留");

  // 4) 上一句还在流、下一句已到：交错更新不能互相踩
  const n35 = nodeOf(35);
  const n36 = nodeOf(36);
  applyUpd([{ seq: 35, trans: "很长的译文".repeat(20) + "，继续", final: false }]);
  applyUpd([{ seq: 36, trans: "下一句起头", final: false }]);
  render();
  ok(nodeOf(35) === n35 && nodeOf(36) === n36, "交错更新不换节点");
  ok(n35.textContent.includes("继续"), "长句后续增量写进去了");
  ok(n36.textContent.includes("下一句起头"), "新句译文也写进去了");

  // 5) pending / final 状态切换：不能把已有译文清空
  applyUpd([{ seq: 36, trans: "下一句起头", final: true, trans_ms: 800, ttft_ms: 300 }]);
  render();
  ok(n36.textContent.includes("下一句起头"), "final 后译文保留");
  const tr36 = dom.get(36).tr;
  ok(tr36.className === "tr", "final 后不再显示光标类名：" + tr36.className);

  // 6) 草稿行：出现在最上（它是最新的一条），且不打断已有行
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

  // 7) 往下翻看旧行时：新行把内容推下去，滚动要等量补回，视线不能跳
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

  // 8) 清屏
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
