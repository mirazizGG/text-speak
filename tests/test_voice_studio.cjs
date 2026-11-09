"use strict";

// Small DOM/fetch harness: exercise user actions and delayed network bodies
// without browser packages, credentials, network access or inference.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../static/voice-studio.js"), "utf8");
const initialVoices = [{ id: "female_1", name: "Existing voice" }];
const savedVoice = { id: "clone_new", name: "New voice", source: "clone" };

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

async function settle() {
  for (let i = 0; i < 4; i++) await new Promise(setImmediate);
}

function studio(respond) {
  const elements = [];
  const downloads = [];
  class Element {
    constructor(tag) {
      this.tag = tag;
      this.children = [];
      this.handlers = new Map();
      this.dataset = {};
      this.value = "";
      this.files = [];
      this.classList = { toggle() {} };
    }
    setAttribute(name, value) { this[name] = value; }
    removeAttribute(name) { delete this[name]; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, listener) {
      const listeners = this.handlers.get(name) || [];
      listeners.push(listener);
      this.handlers.set(name, listeners);
    }
    async emit(name) {
      const event = { target: this, preventDefault() {} };
      for (const listener of this.handlers.get(name) || []) await listener(event);
    }
    pause() {}
    focus() {}
    click() { if (this.tag === "a" && this.href && this.download) downloads.push({ href: this.href, filename: this.download }); return this.emit("click"); }
  }
  const swagger = new Element("div");
  swagger.parentNode = { insertBefore() {} };
  const timers = new Map();
  const intervals = new Map();
  const calls = [];
  const urls = [];
  const revoked = [];
  let credential = "test-only-key";
  let nextTimer = 0;
  const context = {
    document: {
      readyState: "complete",
      createElement(tag) { const element = new Element(tag); elements.push(element); return element; },
      getElementById(id) { return id === "swagger-ui" ? swagger : elements.find(element => element.id === id); },
      addEventListener() {},
      querySelector() { return null; },
    },
    window: { addEventListener() {}, confirm() { return true; } },
    ui: { authSelectors: { authorized() { return { getIn() { return credential; } }; } } },
    URL: { revokeObjectURL(url) { revoked.push(url); }, createObjectURL() { const url = `blob:test-only-${urls.length}`; urls.push(url); return url; } },
    AbortController,
    Element,
    FormData: class { append() {} },
    setTimeout(callback, delay) { const id = ++nextTimer; timers.set(id, { callback, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    setInterval(callback, delay) { const id = ++nextTimer; intervals.set(id, { callback, delay }); return id; },
    clearInterval(id) { intervals.delete(id); },
    async fetch(url, options) { calls.push({ url, options }); return respond(url, options); },
  };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "voice-studio.js" });
  const byId = id => elements.find(element => element.id === `voice-studio-${id}`);
  const descendants = element => [element, ...(element.children || []).flatMap(descendants)];
  return {
    byId, calls, timers, downloads, urls, revoked,
    button(text) { return elements.find(element => element.tag === "button" && element.textContent === text); },
    status() { return elements.find(element => element.role === "status"); },
    jobs() { return descendants(byId("jobs")).filter(element => element.tag === "li"); },
    text(element) { return descendants(element).map(node => node.textContent || "").join(" "); },
    async auth(value) { credential = value; for (const timer of intervals.values()) timer.callback(); await settle(); },
    async tick(delay) {
      const [id, timer] = [...timers].find(([, timer]) => timer.delay === delay) || [];
      assert.ok(timer, `expected a ${delay}ms timer`); timers.delete(id); timer.callback(); await settle();
    },
    async tickAll(delay) {
      const pending = [...timers].filter(([, timer]) => timer.delay === delay);
      assert.ok(pending.length, `expected a ${delay}ms timer`);
      for (const [id, timer] of pending) { timers.delete(id); timer.callback(); }
      await settle();
    },
    async select(id = "female_1") { byId("voice").value = id; await byId("voice").emit("change"); },
    async bulk(content, prefix = "") {
      byId("input-mode").value = "bulk"; await byId("input-mode").emit("change");
      byId("bulk-text").value = content; await byId("bulk-text").emit("input");
      byId("key").value = prefix; byId("speech-language").value = "uz";
      await byId("speak").emit("submit"); await settle();
    },
    fillSave() {
      byId("file").files = [{ size: 10, name: "voice.wav" }];
      byId("name").value = "New voice";
      byId("gender").value = "female";
      byId("reference-language").value = "uz";
    },
  };
}

function json(body) { return { ok: true, status: 200, json: async () => body }; }
function audio(filename = "speech.wav", body = { size: 8 }) {
  return { ok: true, status: 200, headers: { get(name) { return name === "Content-Type" ? "audio/wav" : `attachment; filename="${filename}"`; } }, blob: async () => body };
}
function group(id, items, state = "queued") {
  return { id, jobs: items.map((item, index) => ({ ...item, id: `${id}-${index}`, group_id: id,
    state, filename: `${item.key || "speech"}.wav`, voice_name: item.voice === "male_1" ? "Second voice" : "Existing voice" })) };
}
function jobList(data = [], execution = { parallel_workers: 2, queued: 0, running: 0 }, history) { return json({ data, execution, ...(history ? { history } : {}) }); }
function historyPage(groups, offset = 0, limit = 20) {
  const active = groups.filter(group => group.jobs.some(job => ["queued", "running"].includes(job.state)));
  const completed = groups.filter(group => !active.includes(group));
  return jobList([...active, ...completed.slice(offset, offset + limit)],
    { parallel_workers: 2, queued: active.length, running: 0 },
    { limit, offset, has_more: completed.length > offset + limit });
}

test("a delayed pre-save catalog cannot discard the newly saved selection", async () => {
  const staleBody = deferred();
  let catalogCalls = 0;
  let saved = false;
  const app = studio(async url => {
    if (url === "/v1/audio/jobs") return jobList();
    if (url === "/v1/voices") {
      catalogCalls++;
      if (catalogCalls === 2) return { ok: true, json: () => staleBody.promise };
      return json({ data: saved ? [...initialVoices, savedVoice] : initialVoices });
    }
    assert.equal(url, "/v1/voices/clone");
    saved = true;
    return json(savedVoice);
  });
  await settle();
  app.fillSave();
  const refresh = app.button("Ro‘yxatni yangilash").emit("click");
  await settle();
  await app.byId("save").emit("submit");
  staleBody.resolve({ data: initialVoices });
  await refresh;
  await settle();
  // Either guard saving during catalog refresh, or invalidate that old read.
  // Both approaches must preserve the selection once the save succeeds.
  if (!saved) await app.byId("save").emit("submit");
  await settle();
  assert.equal(app.byId("voice").value, savedVoice.id);
  assert.equal(app.byId("speak").hidden, false);
  assert.equal(app.calls.filter(call => call.url === "/v1/voices/clone").length, 1);
});

test("a long save has no fixed request deadline and logout still aborts its response body", async () => {
  let saveSignal;
  const app = studio(async (url, options) => {
    if (url === "/v1/audio/jobs") return jobList();
    if (url === "/v1/voices") return json({ data: initialVoices });
    assert.equal(url, "/v1/voices/clone");
    saveSignal = options.signal;
    return {
      ok: true,
      json: () => new Promise((resolve, reject) => {
        saveSignal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
      }),
    };
  });
  await settle();
  app.fillSave();
  const saving = app.byId("save").emit("submit");
  await settle();
  const deadlines = [...app.timers.values()].filter(timer => timer.delay === 200000);
  assert.equal(deadlines.length, 0, "long audio must not expire after an arbitrary browser deadline");
  assert.equal(saveSignal.aborted, false);
  assert.equal(app.button("Ovozni saqlash").disabled, true);
  await app.auth(null);
  await saving;
  await settle();
  assert.equal(saveSignal.aborted, true);
  await app.auth("new-test-only-key");
  assert.equal(app.button("Ovozni saqlash").disabled, false);
});

test("100 pasted blocks are queued together and another submission can follow immediately", async () => {
  const accepted = [];
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    assert.equal(url, "/v1/audio/jobs");
    if (options.method !== "POST") return jobList(accepted);
    const body = JSON.parse(options.body);
    const result = group(`group-${accepted.length}`, body.items);
    accepted.unshift(result); return json(result);
  });
  await settle(); await app.select();
  await app.bulk(Array.from({ length: 100 }, (_, i) => `Matn ${i + 1}`).join("\r\n\r\n"), "lesson");
  const submitted = app.calls.filter(call => call.options.method === "POST");
  const items = JSON.parse(submitted[0].options.body).items;
  assert.equal(items.length, 100);
  assert.equal(items[0].key, "lesson.001"); assert.equal(items[99].key, "lesson.100");
  assert.ok(items.every(item => item.voice === "female_1" && item.language === "uz"));
  assert.equal(app.jobs().length, 100);
  assert.equal(app.button("Matnlarni navbatga qo‘shish").disabled, false);
  assert.equal(app.byId("voice").disabled, false);
  assert.equal(app.downloads.length, 0);
  await app.bulk("Yangi birinchi\n\nYangi ikkinchi");
  assert.equal(app.calls.filter(call => call.options.method === "POST").length, 2);
  assert.equal(app.jobs().length, 102);
});

test("JSON bulk input preserves explicit keys and validates every item before sending", async () => {
  let submitted = [];
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (options.method !== "POST") return jobList();
    submitted = JSON.parse(options.body).items; return json(group("json-group", submitted));
  });
  await settle(); await app.select();
  const items = Array.from({ length: 100 }, (_, i) => ({ text: `JSON matn ${i}`, ...(i === 0 ? { key: "custom.first" } : {}) }));
  await app.bulk(JSON.stringify(items));
  assert.equal(submitted.length, 100);
  assert.equal(submitted[0].key, "custom.first"); assert.equal(submitted[99].key, "audio.100");
  assert.equal(app.byId("bulk-count").textContent, "100 ta matn");
  const postCount = () => app.calls.filter(call => call.options.method === "POST").length;
  await app.bulk(JSON.stringify([{ text: "To‘g‘ri" }, { text: "Invalid\u0000text" }]));
  assert.equal(postCount(), 1, "an invalid later item must prevent the entire batch");
  await app.bulk(JSON.stringify([{ text: "One", key: "same" }, { text: "Two", key: "SAME" }]));
  assert.equal(postCount(), 1, "case-insensitive duplicate filenames are rejected");
  await app.bulk(JSON.stringify([{ text: "One", voice: "male_1" }]));
  assert.equal(postCount(), 1, "JSON cannot silently override the visible voice selector");
});

test("long text and more than 1000 bulk entries are sent without content quotas", async () => {
  const submitted = [];
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (options.method !== "POST") return jobList();
    const items = JSON.parse(options.body).items;
    submitted.push(items); return json(group(`unlimited-${submitted.length}`, items));
  });
  await settle(); await app.select();
  assert.equal(app.byId("text").maxlength, undefined);
  assert.equal(app.byId("transcript").maxlength, undefined);
  const longText = "Uzoq matn. ".repeat(2000);
  app.byId("text").value = longText; await app.byId("text").emit("input");
  app.byId("speech-language").value = "uz";
  await app.byId("speak").emit("submit"); await settle();
  assert.equal(submitted[0][0].text, longText.trim());
  const entries = Array.from({ length: 1001 }, (_, i) => ({ text: i ? `Matn ${i}` : longText }));
  await app.bulk(JSON.stringify(entries));
  assert.equal(submitted[1].length, 1001);
  assert.equal(submitted[1][0].text, longText.trim());
  assert.equal(submitted[1][1000].key, "audio.1001");
  assert.equal(app.byId("bulk-count").textContent, "1001 ta matn");
});

test("a large voice upload and transcript are sent without the former 8MiB quota", async () => {
  let saved = false;
  const app = studio(async url => {
    if (url === "/v1/audio/jobs") return jobList();
    if (url === "/v1/voices") return json({ data: saved ? [...initialVoices, savedVoice] : initialVoices });
    assert.equal(url, "/v1/voices/clone"); saved = true; return json(savedVoice);
  });
  await settle(); app.fillSave();
  app.byId("file").files = [{ size: 24 * 1024 * 1024, name: "long-phone-recording.m4a" }];
  app.byId("transcript").value = "Yozuvdagi gap. ".repeat(300);
  await app.byId("save").emit("submit"); await settle();
  assert.equal(saved, true);
  assert.equal(app.byId("voice").value, savedVoice.id);
  assert.equal(app.status().textContent, "Ovoz saqlandi. Endi matn yozing va ovoz yarating.");
});

test("saving a long recording explains the selected voice excerpt and its matching automatic transcript", async () => {
  const excerptVoice = { ...savedVoice, reference_preparation: {
    source_seconds: 120, selected_start_seconds: 32, selected_seconds: 15, automatic_transcript: true,
  } };
  let saved = false;
  const app = studio(async url => {
    if (url === "/v1/audio/jobs") return jobList();
    if (url === "/v1/voices") return json({ data: saved ? [...initialVoices, excerptVoice] : initialVoices });
    assert.equal(url, "/v1/voices/clone"); saved = true; return json(excerptVoice);
  });
  await settle(); app.fillSave();
  app.byId("transcript").value = "To‘liq uzun yozuv uchun oldindan yozilgan matn.";
  await app.byId("save").emit("submit"); await settle();
  assert.equal(app.byId("voice").value, savedVoice.id);
  assert.equal(app.byId("speak").hidden, false);
  // The status must describe what was saved even when the user supplied a
  // transcript for the original full-length recording.
  assert.match(app.status().textContent, /Yozuvdan ovozli namuna tanlandi/);
  assert.match(app.status().textContent, /Namuna matni avtomatik aniqlanib, ishlatildi/);
  assert.match(app.status().textContent, /Endi matn yozing va ovoz yarating/);
});

test("large TXT imports and raw JSON content are not blocked by the former 12MiB quota", async () => {
  const submitted = [];
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (options.method !== "POST") return jobList();
    const items = JSON.parse(options.body).items;
    submitted.push(items); return json(group("large-file", items));
  });
  await settle(); await app.select();
  const content = "a".repeat(12 * 1024 * 1024 + 1);
  app.byId("bulk-file").files = [{ size: content.length, async text() { return content; } }];
  await app.byId("bulk-file").emit("change");
  assert.equal(app.byId("bulk-text").value.length, content.length);
  await app.bulk(JSON.stringify([{ text: content }]));
  assert.equal(submitted[0][0].text.length, content.length);
});

test("polling updates each bulk job independently and only finished audio can be downloaded", async () => {
  let current;
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url.endsWith("/audio")) return audio("audio.001.wav");
    assert.equal(url, "/v1/audio/jobs");
    if (options.method === "POST") { current = group("mixed", JSON.parse(options.body).items); return json(current); }
    return jobList(current ? [current] : [], { parallel_workers: 2, running: current ? 1 : 0, queued: current ? 1 : 0 });
  });
  await settle(); await app.select(); await app.bulk("A\n\nB\n\nC\n\nD");
  current = { ...current, jobs: current.jobs.map((job, i) => ({ ...job, state: ["succeeded", "running", "queued", "failed"][i] })) };
  await app.tick(1000);
  const rows = app.jobs();
  assert.deepEqual(rows.map(row => row.dataset.state), ["succeeded", "running", "queued", "failed"]);
  assert.match(app.text(rows[0]), /Tayyor/); assert.match(app.text(rows[3]), /Xatolik/);
  assert.equal(rows[0].children[2].children[1].hidden, false);
  assert.equal(rows[1].children[2].children[1].hidden, true);
  assert.match(app.byId("queue-summary").textContent, /Bir vaqtda: 2 ta.*Jarayonda: 1.*Navbatda: 1/);
  assert.equal(app.downloads.length, 0, "bulk polling never triggers automatic downloads");
  await rows[0].children[2].children[1].emit("click"); await settle();
  assert.equal(app.downloads.length, 1); assert.equal(app.downloads[0].filename, "audio.001.wav");
  assert.equal(app.calls.at(-1).options.headers.Authorization, "Bearer test-only-key");
  await app.tick(1000);
  // The download URL may be revoked by the following timer after the poll.
  if (!app.revoked.includes(app.downloads[0].href)) await app.tick(1000);
  assert.ok(app.revoked.includes(app.downloads[0].href));
});

test("logout rejects a delayed jobs body and a new login restores protected history", async () => {
  const delayed = deferred(); let reads = 0;
  const previous = group("previous", [{ text: "Saqlangan matn", voice: "female_1", key: "old" }], "succeeded");
  const app = studio(async url => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    assert.equal(url, "/v1/audio/jobs"); reads++;
    if (reads === 1) return { ok: true, json: () => delayed.promise };
    return jobList([previous]);
  });
  await settle(); await app.auth(null);
  delayed.resolve({ data: [previous] }); await settle();
  assert.equal(app.jobs().length, 0); assert.equal(app.byId("queue").hidden, true);
  assert.equal([...app.timers.values()].filter(timer => timer.delay <= 5000).length, 0);
  await app.auth("new-test-only-key");
  assert.equal(app.jobs().length, 1); assert.equal(app.byId("queue").hidden, false);
  assert.equal(app.downloads.length, 0, "restoring a completed group does not download it automatically");
});

test("a single job previews without downloading and its separate download keeps the submitted voice", async () => {
  let current;
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: [...initialVoices, { id: "male_1", name: "Second voice" }] });
    if (url.endsWith("/audio")) return audio("welcome.wav");
    if (options.method === "POST") { current = group("single", JSON.parse(options.body).items); return json(current); }
    return jobList(current ? [current] : []);
  });
  await settle(); await app.select();
  assert.equal(app.byId("download").hidden, true);
  assert.equal(app.byId("download").disabled, true);
  app.byId("text").value = "Salom"; app.byId("key").value = "welcome"; app.byId("speech-language").value = "uz";
  await app.byId("speak").emit("submit"); await settle();
  assert.equal(app.button("Ovozni yaratish").disabled, false);
  assert.equal(app.byId("download").hidden, true);
  await app.select("male_1");
  current = { ...current, jobs: current.jobs.map(job => ({ ...job, state: "succeeded" })) };
  await app.tick(1000);
  assert.equal(app.byId("voice").value, "male_1");
  assert.equal(app.byId("output-label").textContent, "welcome.wav · Existing voice");
  assert.equal(app.downloads.length, 0, "finishing generation never saves a file automatically");
  assert.equal(app.byId("download").hidden, false);
  assert.equal(app.byId("download").disabled, false);
  assert.equal(app.byId("download").textContent, "Yuklab olish");
  const actions = app.byId("speak").children.find(element => element.children.includes(app.button("Ovozni yaratish")));
  assert.ok(actions.children.includes(app.byId("download")), "download appears beside create");
  await app.byId("download").emit("click"); await settle();
  assert.equal(app.downloads.length, 1); assert.equal(app.downloads[0].filename, "welcome.wav");
  await app.byId("download").emit("click"); await settle();
  assert.equal(app.downloads.length, 2, "another explicit click can download again");
  assert.equal(app.downloads[1].filename, "welcome.wav");
  assert.match(app.text(app.jobs()[0]), /Existing voice/);
  await app.select("female_1");
  assert.equal(app.byId("output-label").hidden, false);
  await app.auth(null);
  assert.equal(app.byId("output-label").hidden, true); assert.equal(app.jobs().length, 0);
  assert.equal(app.byId("download").hidden, true); assert.equal(app.byId("download").disabled, true);
  assert.ok(app.urls.every(url => app.revoked.includes(url)), "logout revokes playback and outstanding download URLs");
});

test("manual download suppresses duplicate clicks while its audio is loading", async () => {
  const delayed = deferred(); let current; let audioReads = 0;
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url.endsWith("/audio")) {
      audioReads += 1;
      const result = audio("manual.wav");
      if (audioReads > 1) result.blob = () => delayed.promise;
      return result;
    }
    if (options.method === "POST") { current = group("manual", JSON.parse(options.body).items); return json(current); }
    return jobList(current ? [current] : []);
  });
  await settle(); await app.select();
  app.byId("text").value = "Salom"; app.byId("key").value = "manual";
  await app.byId("speak").emit("submit");
  current = { ...current, jobs: current.jobs.map(job => ({ ...job, state: "succeeded" })) };
  await app.tick(1000);
  assert.equal(audioReads, 1); assert.equal(app.downloads.length, 0);
  await app.byId("download").emit("click");
  await app.byId("download").emit("click"); await settle();
  assert.equal(audioReads, 2, "only the first pending click fetches audio");
  assert.equal(app.byId("download").disabled, true); assert.equal(app.downloads.length, 0);
  delayed.resolve({ size: 8 }); await settle();
  assert.equal(app.downloads.length, 1); assert.equal(app.downloads[0].filename, "manual.wav");
  assert.equal(app.byId("download").disabled, false);
  await app.tickAll(1000);
  assert.equal(app.downloads.length, 1, "later polling cannot download the result again");
});

test("logout during a manual download cannot save the delayed audio", async () => {
  const delayed = deferred(); let audioReads = 0;
  const previous = group("manual-logout", [{ text: "Matn", voice: "female_1" }], "succeeded");
  const app = studio(async url => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url === "/v1/audio/jobs") return jobList([previous]);
    audioReads += 1; const result = audio();
    if (audioReads > 1) result.blob = () => delayed.promise;
    return result;
  });
  await settle(); await app.jobs()[0].children[2].children[0].emit("click"); await settle();
  await app.byId("download").emit("click"); await settle();
  await app.auth(null); delayed.resolve({ size: 8 }); await settle();
  assert.equal(app.downloads.length, 0); assert.equal(app.jobs().length, 0);
  assert.equal(app.byId("download").hidden, true); assert.equal(app.byId("download").disabled, true);
  assert.ok(app.urls.every(url => app.revoked.includes(url)));
});

test("a stale audio response cannot repopulate playback after logout", async () => {
  const delayed = deferred();
  const previous = group("old-audio", [{ text: "Matn", voice: "female_1" }], "succeeded");
  const app = studio(async url => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url === "/v1/audio/jobs") return jobList([previous]);
    const response = audio(); response.blob = () => delayed.promise; return response;
  });
  await settle();
  const playing = app.jobs()[0].children[2].children[0].emit("click"); await settle();
  await app.auth(null); delayed.resolve({ size: 8 }); await playing; await settle();
  assert.equal(app.urls.length, 0); assert.equal(app.byId("output-label").hidden, true);
  assert.equal(app.byId("download").hidden, true); assert.equal(app.byId("download").disabled, true);
  assert.equal(app.downloads.length, 0);
  assert.equal(app.jobs().length, 0);
});

test("a 401 while polling clears protected history and playback and stops polling", async () => {
  let rejected = false;
  const previous = group("protected", [{ text: "Matn", voice: "female_1" }], "succeeded");
  const app = studio(async url => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url.endsWith("/audio")) return audio();
    return rejected ? { ok: false, status: 401 } : jobList([previous]);
  });
  await settle(); await app.jobs()[0].children[2].children[0].emit("click"); await settle();
  assert.equal(app.byId("output-label").hidden, false);
  rejected = true; await app.tick(5000);
  assert.equal(app.jobs().length, 0); assert.equal(app.byId("queue").hidden, true);
  assert.equal(app.byId("output-label").hidden, true);
  assert.equal(app.byId("download").hidden, true); assert.equal(app.byId("download").disabled, true);
  assert.equal(app.byId("voice").disabled, true);
  assert.ok(app.urls.every(url => app.revoked.includes(url)));
  assert.equal([...app.timers.values()].filter(timer => timer.delay <= 5000).length, 0);
});

test("existing saved voice sample and deletion remain available beside the queue", async () => {
  let deleted = false;
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: deleted ? initialVoices : [...initialVoices, savedVoice] });
    if (url === "/v1/audio/jobs") return jobList();
    if (url === `/v1/voices/${savedVoice.id}/sample`) return audio();
    assert.equal(url, `/v1/voices/${savedVoice.id}`); assert.equal(options.method, "DELETE");
    deleted = true; return json({ deleted: true });
  });
  await settle(); await app.select(savedVoice.id);
  assert.equal(app.button("Ovozni o‘chirish").disabled, false);
  await app.button("Ovozni eshitish").emit("click"); await settle();
  assert.equal(app.urls.length, 1);
  await app.button("Ovozni o‘chirish").emit("click"); await settle();
  assert.equal(deleted, true); assert.equal(app.byId("voice").value, "");
  assert.ok(app.revoked.includes(app.urls[0]));
  assert.equal(app.button("Ovozni saqlash").disabled, false);
});

test("50 separate submissions retain and finish the oldest job beyond the recent history page", async () => {
  const groups = [];
  const app = studio(async (url, options) => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url === "/v1/audio/jobs" && options.method === "POST") {
      const accepted = group(`separate-${groups.length}`, JSON.parse(options.body).items);
      groups.unshift(accepted); return json(accepted);
    }
    if (url === "/v1/audio/jobs") return historyPage(groups);
    const found = groups.find(group => url.startsWith(`/v1/audio/jobs/${group.id}`));
    assert.ok(found, `unexpected route ${url}`);
    return url.endsWith("/audio") ? audio(found.jobs[0].filename) : json(found);
  });
  await settle(); await app.select(); app.byId("speech-language").value = "uz";
  for (let i = 0; i < 50; i++) {
    app.byId("text").value = `Alohida matn ${i}`; app.byId("key").value = `separate.${i}`;
    await app.byId("speak").emit("submit");
  }
  await settle();
  const first = groups.at(-1);
  assert.equal(app.jobs().length, 50); assert.equal(app.downloads.length, 0);
  // Forty-nine later requests finish while the oldest one is still pending.
  for (const later of groups.slice(0, -1)) later.jobs = later.jobs.map(job => ({ ...job, state: "succeeded" }));
  await app.tick(1000);
  assert.equal(app.jobs().length, 50);
  assert.equal(app.jobs().find(row => row.dataset.jobId === first.jobs[0].id).dataset.state, "queued");
  first.jobs = first.jobs.map(job => ({ ...job, state: "succeeded" }));
  // Its completion is outside the newest 20 terminal groups, so it needs its own GET.
  await app.tickAll(1000);
  const completed = app.jobs().find(row => row.dataset.jobId === first.jobs[0].id);
  assert.equal(completed.dataset.state, "succeeded");
  assert.equal(completed.children[2].children[1].hidden, false);
  assert.ok(app.calls.some(call => call.url === `/v1/audio/jobs/${first.id}`));
  assert.equal(app.downloads.length, 0, "completing single jobs outside the recent page never downloads them");
  await completed.children[2].children[1].emit("click"); await settle();
  assert.equal(app.downloads.filter(download => download.filename === "separate.0.wav").length, 1);
  assert.equal(app.jobs().length, 50);
  await app.tick(5000);
  assert.equal(app.downloads.filter(download => download.filename === "separate.0.wav").length, 1);
});

test("login restores all pending groups and older completed pages merge without duplicates", async () => {
  const groups = [
    ...Array.from({ length: 50 }, (_, i) => group(`pending-${i}`, [{ text: `Pending ${i}`, voice: "female_1" }])),
    ...Array.from({ length: 45 }, (_, i) => group(`done-${i}`, [{ text: `Done ${i}`, voice: "female_1", key: `done.${i}` }], "succeeded")),
  ];
  const app = studio(async url => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    const parsed = new URL(url, "http://local.test");
    assert.equal(parsed.pathname, "/v1/audio/jobs");
    return historyPage(groups, Number(parsed.searchParams.get("offset") || 0));
  });
  await settle();
  assert.equal(app.jobs().length, 70, "all 50 active groups are restored along with the first 20 results");
  assert.equal(app.byId("history").hidden, false);
  await app.byId("history").emit("click"); await settle();
  assert.equal(app.jobs().length, 90);
  await app.byId("history").emit("click"); await settle();
  assert.equal(app.jobs().length, 95); assert.equal(app.byId("history").hidden, true);
  assert.equal(new Set(app.jobs().map(row => row.dataset.jobId)).size, 95);
  await app.tick(1000);
  assert.equal(app.jobs().length, 95, "polling the first page retains explicitly loaded history");
  await app.auth(null); assert.equal(app.jobs().length, 0); assert.equal(app.byId("history").hidden, true);
  await app.auth("new-test-only-key");
  assert.equal(app.jobs().length, 70);
  await app.byId("history").emit("click"); await settle();
  assert.equal(app.jobs().length, 90); assert.equal(app.downloads.length, 0);
  assert.ok(app.calls.some(call => call.url === "/v1/audio/jobs?limit=20&offset=20"));
  assert.ok(app.calls.some(call => call.url === "/v1/audio/jobs?limit=20&offset=40"));
});

test("an old history page cannot repopulate a new login", async () => {
  const delayed = deferred();
  const recent = Array.from({ length: 20 }, (_, i) => group(`recent-${i}`, [{ text: `Recent ${i}`, voice: "female_1" }], "succeeded"));
  const protectedOld = group("old-private", [{ text: "Previous login", voice: "female_1" }], "succeeded");
  const app = studio(async url => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url.includes("offset=20")) return { ok: true, json: () => delayed.promise };
    return jobList(recent, undefined, { limit: 20, offset: 0, has_more: true });
  });
  await settle(); const loading = app.byId("history").emit("click"); await settle();
  await app.auth(null); await app.auth("new-test-only-key");
  delayed.resolve({ data: [protectedOld], history: { limit: 20, offset: 20, has_more: false } });
  await loading; await settle();
  assert.equal(app.jobs().length, 20);
  assert.equal(app.jobs().some(row => row.dataset.jobId === protectedOld.jobs[0].id), false);
  assert.equal(app.byId("history").disabled, false);
});

test("a delayed history snapshot cannot regress a completed job to queued", async () => {
  const delayed = deferred();
  const queued = group("progress", [{ text: "Progress", voice: "female_1" }]);
  const finished = { ...queued, jobs: queued.jobs.map(job => ({ ...job, state: "succeeded" })) };
  let current = queued;
  const app = studio(async url => {
    if (url === "/v1/voices") return json({ data: initialVoices });
    if (url.includes("offset=20")) return { ok: true, json: () => delayed.promise };
    return jobList([current], undefined, { limit: 20, offset: 0, has_more: true });
  });
  await settle(); const loading = app.byId("history").emit("click"); await settle();
  current = finished; await app.tick(1000);
  assert.equal(app.jobs()[0].dataset.state, "succeeded");
  delayed.resolve({ data: [queued], history: { limit: 20, offset: 20, has_more: false } });
  await loading; await settle();
  assert.equal(app.jobs().length, 1); assert.equal(app.jobs()[0].dataset.state, "succeeded");
  assert.equal(app.jobs()[0].children[2].children[1].hidden, false);
});
