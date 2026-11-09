(function () {
  "use strict";

  function element(tag, attributes, ...children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attributes || {})) {
      if (name === "className") node.className = value;
      else if (name === "text") node.textContent = value;
      else if (name === "hidden") node.hidden = value;
      else if (name === "htmlFor") node.htmlFor = value;
      else if (name === "required") node.required = value;
      else node.setAttribute(name, String(value));
    }
    node.append(...children.flat());
    return node;
  }

  function field(label, input, hint) {
    const children = [element("label", { htmlFor: input.id, text: label }), input];
    if (hint) children.push(element("small", { text: hint }));
    return element("div", { className: "voice-studio__field" }, children);
  }

  function mount() {
    const swagger = document.getElementById("swagger-ui");
    if (!swagger || document.getElementById("voice-studio")) return;

    const state = { token: null, rejectedToken: null, authVersion: 0, voices: [], busy: false, loading: false, refreshPending: false,
      selected: "", catalogVersion: 0, sampleVersion: 0, sampleUrl: null, outputUrl: null, outputJob: null,
      groups: [], groupViews: new Map(), groupOrder: "", groupsVersion: 0, groupsLoadId: 0, groupsLoading: false, pollTimer: null, execution: null,
      historyNextOffset: 0, historyMore: false, historyLoading: false, historyLoadId: 0,
      previewGroups: new Map(), transfers: new Set(), downloadUrls: new Set(), controllers: new Set() };
    const title = element("h1", { id: "voice-studio-title", text: "Ovoz studiyasi" });
    const subtitle = element("p", { className: "voice-studio__subtitle", text: "Ovozingizni saqlang, matn yozing va tayyor audioni yuklab oling." });
    const authButton = element("button", { type: "button", className: "voice-studio__button voice-studio__button--outline", text: "Authorize orqali kirish" });
    const access = element("span", { className: "voice-studio__access", text: "Kirish talab qilinadi" });
    const status = element("div", { className: "voice-studio__status", role: "status", "aria-live": "polite", text: "Boshlash uchun Authorize orqali kiring." });

    const voices = element("select", { id: "voice-studio-voice", "aria-describedby": "voice-studio-voice-help" });
    const voiceHelp = element("small", { id: "voice-studio-voice-help", text: "Oldin saqlagan ovozingizni yoki tayyor ovozni tanlang." });
    const voiceField = element("div", { className: "voice-studio__field" }, element("label", { htmlFor: voices.id, text: "Ovoz tanlang" }), voices, voiceHelp);
    const file = element("input", { id: "voice-studio-file", type: "file", accept: "audio/*,.m4a,.aac,.ogg,.opus,.webm,.amr,.3gp,.wma,.aiff,.caf,.mp4,.mkv,.avi" });
    const fileField = field("Yangi ovoz yozuvi", file, "Audio fayllar avtomatik tayyorlanadi. Uzun yozuvdan ovozli namuna tanlanadi.");
    const refreshButton = element("button", { type: "button", className: "voice-studio__button voice-studio__button--soft", text: "Ro‘yxatni yangilash" });
    const sampleButton = element("button", { type: "button", className: "voice-studio__button voice-studio__button--soft", text: "Ovozni eshitish" });
    const deleteButton = element("button", { type: "button", className: "voice-studio__button voice-studio__button--danger", text: "Ovozni o‘chirish" });
    const sampleAudio = element("audio", { controls: "", hidden: true, "aria-label": "Tanlangan ovoz namunasi" });
    const selectorActions = element("div", { className: "voice-studio__actions" }, refreshButton, sampleButton, deleteButton);

    const saveTab = element("button", { type: "button", role: "tab", id: "voice-studio-tab-save", "aria-controls": "voice-studio-save", "aria-selected": "true", tabindex: "0", className: "voice-studio__tab is-active", text: "1. Ovoz saqlash" });
    const speakTab = element("button", { type: "button", role: "tab", id: "voice-studio-tab-speak", "aria-controls": "voice-studio-speak", "aria-selected": "false", tabindex: "-1", className: "voice-studio__tab", text: "2. Matn yozish" });
    const tabs = element("div", { role: "tablist", "aria-label": "Ovoz bilan ishlash", className: "voice-studio__tabs" }, saveTab, speakTab);

    const name = element("input", { id: "voice-studio-name", type: "text", maxlength: "80", placeholder: "Masalan: Mening ovozim", required: true });
    const gender = element("select", { id: "voice-studio-gender", required: true },
      element("option", { value: "", text: "Tanlang" }),
      element("option", { value: "female", text: "Ayol ovozi" }),
      element("option", { value: "male", text: "Erkak ovozi" }));
    const referenceLanguage = element("select", { id: "voice-studio-reference-language" },
      element("option", { value: "uz", text: "O‘zbekcha" }),
      element("option", { value: "ru", text: "Ruscha" }),
      element("option", { value: "en", text: "Inglizcha" }));
    const transcript = element("textarea", { id: "voice-studio-transcript", rows: "2", placeholder: "Yozuvdagi gapni bilsangiz, shu yerga yozing" });
    const saveButton = element("button", { type: "submit", className: "voice-studio__button voice-studio__button--primary", text: "Ovozni saqlash" });
    const saveForm = element("form", { id: "voice-studio-save", role: "tabpanel", "aria-labelledby": saveTab.id, className: "voice-studio__panel" },
      element("p", { className: "voice-studio__intro", text: "Yangi ovoz yozuvini tanlang va unga nom bering." }),
      element("div", { className: "voice-studio__grid voice-studio__grid--three" },
        field("Ovoz nomi", name), field("Ovoz turi", gender), field("Yozuv tili", referenceLanguage)),
      field("Yozuvdagi gap (ixtiyoriy)", transcript, "Bo‘sh qoldirsangiz, gap avtomatik aniqlanadi."),
      element("div", { className: "voice-studio__actions" }, saveButton));

    const speechText = element("textarea", { id: "voice-studio-text", rows: "4", placeholder: "Ovoz aytishi kerak bo‘lgan matnni yozing", required: true });
    const counter = element("small", { className: "voice-studio__counter", text: "0 belgi" });
    const textField = element("div", { className: "voice-studio__field" }, element("label", { htmlFor: speechText.id, text: "Matn" }), speechText, counter);
    const inputMode = element("select", { id: "voice-studio-input-mode" },
      element("option", { value: "single", text: "Bitta matn" }), element("option", { value: "bulk", text: "Ko‘p matn" }));
    const bulkText = element("textarea", { id: "voice-studio-bulk-text", rows: "6", placeholder: "Birinchi matn\n\nIkkinchi matn\n\nUchinchi matn" });
    const bulkCounter = element("small", { id: "voice-studio-bulk-count", className: "voice-studio__counter", text: "0 ta matn" });
    const bulkFile = element("input", { id: "voice-studio-bulk-file", type: "file", accept: ".json,.txt,application/json,text/plain" });
    const bulkField = element("div", { className: "voice-studio__field", hidden: true },
      element("label", { htmlFor: bulkText.id, text: "Matnlar" }), bulkText, bulkCounter,
      element("small", { text: 'Matnlarni bo‘sh qator bilan ajrating yoki [{"text":"Salom","key":"salom"}] ko‘rinishida kiriting. Barcha matnlarga tanlangan ovoz va til qo‘llanadi.' }),
      element("label", { htmlFor: bulkFile.id, text: "TXT yoki JSON fayldan ochish" }), bulkFile);
    const speechLanguage = element("select", { id: "voice-studio-speech-language" },
      element("option", { value: "uz", text: "O‘zbekcha" }),
      element("option", { value: "ru", text: "Ruscha" }),
      element("option", { value: "en", text: "Inglizcha" }));
    const key = element("input", { id: "voice-studio-key", type: "text", maxlength: "128", placeholder: "Masalan: assistant.welcome" });
    const filenameHelp = element("small", { text: "Bo‘sh bo‘lsa: speech.wav" });
    const keyField = element("div", { className: "voice-studio__field" }, element("label", { htmlFor: key.id, text: "Fayl nomi (key)" }), key, filenameHelp);
    const generateButton = element("button", { type: "submit", className: "voice-studio__button voice-studio__button--primary", text: "Ovozni yaratish" });
    const outputAudio = element("audio", { controls: "", hidden: true, "aria-label": "Yaratilgan audio" });
    const outputLabel = element("p", { id: "voice-studio-output-label", className: "voice-studio__intro", hidden: true });
    const download = element("button", { id: "voice-studio-download", type: "button", className: "voice-studio__button voice-studio__button--outline", hidden: true, disabled: true, text: "Yuklab olish" });
    const downloadLink = element("a", { hidden: true });
    const queueSummary = element("p", { id: "voice-studio-queue-summary", className: "voice-studio__intro", "aria-live": "polite" });
    const queueList = element("div", { id: "voice-studio-jobs", className: "voice-studio__jobs" });
    const historyButton = element("button", { id: "voice-studio-history", type: "button", className: "voice-studio__button voice-studio__button--soft", text: "Oldingi natijalar", hidden: true });
    const queuePanel = element("div", { id: "voice-studio-queue", className: "voice-studio__queue", hidden: true },
      element("h2", { text: "Matnlar va tayyor audiolar" }), queueSummary, queueList,
      element("div", { className: "voice-studio__actions" }, historyButton));
    const speakForm = element("form", { id: "voice-studio-speak", role: "tabpanel", "aria-labelledby": speakTab.id, className: "voice-studio__panel", hidden: true },
      element("p", { className: "voice-studio__intro", text: "Yuqoridan ovozni tanlang, keyin matn kiriting." }), field("Kiritish usuli", inputMode), textField, bulkField,
      element("div", { className: "voice-studio__grid" }, field("Matn tili", speechLanguage), keyField),
      element("div", { className: "voice-studio__actions" }, generateButton, download), outputLabel, outputAudio);

    const root = element("section", { id: "voice-studio", className: "voice-studio", "aria-labelledby": title.id },
      element("div", { className: "voice-studio__header" }, element("div", null, element("span", { className: "voice-studio__eyebrow", text: "TEXT SPEAK" }), title, subtitle), element("div", { className: "voice-studio__account" }, access, authButton)),
      status, element("div", { className: "voice-studio__grid voice-studio__source-grid" }, voiceField, fileField), selectorActions, sampleAudio, tabs, saveForm, speakForm, queuePanel, downloadLink);
    swagger.parentNode.insertBefore(root, swagger);

    function notify(message, kind) {
      status.textContent = message;
      status.dataset.kind = kind || "info";
    }

    function token() {
      try {
        if (typeof ui === "undefined" || !ui.authSelectors) return null;
        const value = ui.authSelectors.authorized().getIn(["HTTPBearer", "value"]);
        return typeof value === "string" && value ? value : null;
      } catch (_) { return null; }
    }

    function clearSample() {
      state.sampleVersion += 1;
      sampleAudio.pause();
      sampleAudio.removeAttribute("src");
      sampleAudio.hidden = true;
      if (state.sampleUrl) URL.revokeObjectURL(state.sampleUrl);
      state.sampleUrl = null;
    }

    function clearOutput() {
      outputAudio.pause();
      outputAudio.removeAttribute("src");
      outputAudio.hidden = true;
      outputLabel.hidden = true;
      outputLabel.textContent = "";
      download.hidden = true;
      download.disabled = true;
      if (state.outputUrl) URL.revokeObjectURL(state.outputUrl);
      state.outputUrl = null;
      state.outputJob = null;
    }

    function clearJobs() {
      state.groupsVersion += 1;
      if (state.pollTimer !== null) clearTimeout(state.pollTimer);
      state.pollTimer = null; state.groupsLoading = false; state.groups = []; state.execution = null;
      state.groupsLoadId += 1; state.groupViews.clear(); state.groupOrder = "";
      state.historyNextOffset = 0; state.historyMore = false; state.historyLoading = false; state.historyLoadId += 1;
      historyButton.hidden = true;
      state.previewGroups.clear(); state.transfers.clear(); queueList.replaceChildren(); queuePanel.hidden = true;
      queueSummary.textContent = "";
      for (const url of state.downloadUrls) URL.revokeObjectURL(url);
      state.downloadUrls.clear(); downloadLink.removeAttribute("href");
      for (const controller of state.controllers) controller.abort();
    }

    function currentVoice() { return state.voices.find((voice) => voice.id === state.selected); }
    function canUse() { return Boolean(state.token && state.token !== state.rejectedToken); }
    function isCurrent(credential, version) {
      return Boolean(credential && credential === state.token && credential === token() &&
        credential !== state.rejectedToken && version === state.authVersion);
    }

    function updateControls() {
      const active = canUse();
      const voice = currentVoice();
      const locked = state.busy;
      voices.disabled = !active || locked || state.loading;
      refreshButton.disabled = !active || locked || state.loading;
      file.disabled = locked;
      for (const control of [name, gender, referenceLanguage, transcript, speechText, bulkText, bulkFile, inputMode, speechLanguage, key, saveTab, speakTab]) control.disabled = locked;
      authButton.disabled = locked;
      saveButton.disabled = !active || locked || state.loading;
      generateButton.disabled = !active || locked || state.loading || !voice;
      sampleButton.disabled = !active || locked || state.loading || !voice;
      deleteButton.disabled = !active || locked || state.loading || !voice || !["clone", "design"].includes(voice.source);
      deleteButton.title = voice && !["clone", "design"].includes(voice.source) ? "Tayyor ovozlarni o‘chirib bo‘lmaydi" : "";
      access.textContent = active ? "Ulangan" : "Kirish talab qilinadi";
      access.dataset.connected = active ? "true" : "false";
    }

    function activate(which) {
      const saving = which === "save";
      saveForm.hidden = !saving;
      speakForm.hidden = saving;
      saveTab.classList.toggle("is-active", saving);
      speakTab.classList.toggle("is-active", !saving);
      saveTab.setAttribute("aria-selected", String(saving));
      speakTab.setAttribute("aria-selected", String(!saving));
      saveTab.tabIndex = saving ? 0 : -1;
      speakTab.tabIndex = saving ? -1 : 0;
    }

    function populateVoices(items, selected) {
      voices.replaceChildren(element("option", { value: "", text: "Ovozni tanlang" }));
      const saved = element("optgroup", { label: "Saqlangan ovozlar" });
      const builtins = element("optgroup", { label: "Tayyor ovozlar" });
      for (const voice of items) {
        if (!voice || typeof voice.id !== "string" || typeof voice.name !== "string") continue;
        const option = element("option", { value: voice.id, text: voice.name });
        (["clone", "design"].includes(voice.source) ? saved : builtins).append(option);
      }
      if (saved.children.length) voices.append(saved);
      if (builtins.children.length) voices.append(builtins);
      state.voices = items.filter((voice) => voice && typeof voice.id === "string" && typeof voice.name === "string");
      state.selected = state.voices.some((voice) => voice.id === selected) ? selected : "";
      voices.value = state.selected;
      updateControls();
    }

    function setBusy(value) {
      state.busy = value;
      updateControls();
      if (!value && state.refreshPending) {
        state.refreshPending = false;
        refresh(state.selected, true);
      }
    }

    function errorMessage(error, action) {
      if (error && error.status === 401) return "Kirish muddati tugagan yoki kalit noto‘g‘ri. Authorize orqali qayta kiring.";
      if (error && error.status === 403) return "Bu tayyor ovozni o‘chirib bo‘lmaydi.";
      if (error && error.status === 404) return "Ovoz topilmadi. Ro‘yxatni yangilang.";
      if (error && error.status === 413) return "Server so‘rovni qabul qilish uchun yetarli resurs ajrata olmadi.";
      if (error && error.status === 422 && ["speech", "enqueue"].includes(action)) return "Ovoz yoki matn mos kelmadi. Ovozlar ro‘yxatini yangilang va matnni tekshiring.";
      if (error && error.status === 422) return "Audio fayl ochilishini va barcha maydonlar to‘g‘ri to‘ldirilganini tekshiring.";
      if (error && error.status === 429) return "Server band. Birozdan keyin qayta urinib ko‘ring.";
      if (action === "save" && (!error || !error.status || error.status === 504)) return "Saqlash natijasi noma’lum. Qayta saqlashdan oldin ovozlar ro‘yxatini yangilab tekshiring.";
      if (action === "enqueue" && (!error || !error.status || error.status === 504)) return "Navbatga qo‘shish natijasi noma’lum. Qayta yuborishdan oldin matnlar ro‘yxatini tekshiring.";
      if (action === "delete" && (!error || !error.status || error.status === 504 || error.status === 503)) return "O‘chirish natijasini ro‘yxatni yangilab tekshiring. Zarur bo‘lsa, shu ID bilan qayta urinib ko‘ring.";
      if (error && error.status === 503) return "Server yoki avtomatik transkripsiya hozir tayyor emas. Yozuvdagi gapni qo‘lda kiritib ham urinishingiz mumkin.";
      if (error && error.status === 504) return "So‘rov vaqti tugadi. Birozdan keyin qayta urinib ko‘ring.";
      return "Ulanishda xatolik yuz berdi. Qayta urinib ko‘ring.";
    }

    function validKey(value) {
      return value.length <= 128 && /^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$/.test(value) &&
        !/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(value);
    }

    function bulkItems(value) {
      const raw = value.trim();
      if (!raw) return [];
      let items;
      if (raw.startsWith("[")) {
        try { items = JSON.parse(raw); } catch (_) { throw new Error("JSON yozuvini tekshiring: text va ixtiyoriy key maydonlaridan foydalaning."); }
        if (!Array.isArray(items) || items.some((item) => !item || typeof item !== "object" || Array.isArray(item) ||
          typeof item.text !== "string" || (item.key !== undefined && typeof item.key !== "string") ||
          Object.keys(item).some((field) => !["text", "key"].includes(field)))) {
          throw new Error("JSON ro‘yxatidagi har bir yozuv text va ixtiyoriy key maydonlaridan iborat bo‘lsin.");
        }
      } else items = raw.split(/\r?\n[\t ]*\r?\n(?:[\t ]*\r?\n)*/).map((text) => ({ text }));
      return items.map((item, index) => {
        const text = item.text.trim();
        if (!text || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/.test(text)) {
          throw new Error(`${index + 1}-matnni tekshiring: bo‘sh bo‘lmasin va to‘g‘ri belgilar bilan yozilsin.`);
        }
        const itemKey = item.key === undefined ? "" : item.key.trim();
        if (item.key !== undefined && !validKey(itemKey)) throw new Error(`${index + 1}-matnning fayl nomini tekshiring.`);
        return { text, key: itemKey };
      });
    }

    function updateInputMode() {
      const bulk = inputMode.value === "bulk";
      textField.hidden = bulk; bulkField.hidden = !bulk; speechText.required = !bulk; bulkText.required = bulk;
      generateButton.textContent = bulk ? "Matnlarni navbatga qo‘shish" : "Ovozni yaratish";
      filenameHelp.textContent = bulk ? `${key.value.trim() || "audio"}.001.wav, .002.wav, …` : key.value.trim() ? `${key.value.trim()}.wav` : "Bo‘sh bo‘lsa: speech.wav";
    }

    function countJobs(group) {
      const counts = { queued: 0, running: 0, succeeded: 0, failed: 0 };
      for (const job of group.jobs) if (Object.hasOwn(counts, job.state)) counts[job.state] += 1;
      return counts;
    }

    function validGroup(group) {
      return Boolean(group && typeof group.id === "string" && Array.isArray(group.jobs) && group.jobs.every((job) =>
        job && typeof job.id === "string" && typeof job.text === "string" && ["queued", "running", "succeeded", "failed"].includes(job.state)));
    }

    function pendingGroup(group) { return group.jobs.some((job) => ["queued", "running"].includes(job.state)); }

    function mergeGroups(incoming) {
      const groups = new Map(state.groups.map((group) => [group.id, group]));
      for (const group of incoming) {
        const previous = groups.get(group.id);
        if (!previous) { groups.set(group.id, group); continue; }
        const previousJobs = new Map(previous.jobs.map((job) => [job.id, job]));
        const jobs = group.jobs.map((job) => {
          const old = previousJobs.get(job.id);
          // A slower history/poll response must not turn a finished job back into a waiter.
          return old && ["succeeded", "failed"].includes(old.state) && ["queued", "running"].includes(job.state) ? old : job;
        });
        groups.set(group.id, { ...group, jobs });
      }
      state.groups = [...groups.values()];
    }

    function readHistory(body, olderPage) {
      const history = body.history;
      if (!history || !Number.isInteger(history.limit) || history.limit < 1 ||
          !Number.isInteger(history.offset) || history.offset < 0 || typeof history.has_more !== "boolean") return;
      if (olderPage || state.historyNextOffset <= history.limit) {
        state.historyNextOffset = history.offset + history.limit;
        state.historyMore = history.has_more;
      }
    }

    function saveBlob(blob, filename) {
      const url = URL.createObjectURL(blob);
      state.downloadUrls.add(url); downloadLink.href = url; downloadLink.download = filename; downloadLink.click();
      downloadLink.removeAttribute("href");
      setTimeout(() => { if (state.downloadUrls.delete(url)) URL.revokeObjectURL(url); }, 1000);
    }

    async function transferJob(groupId, job, play, save) {
      if (!canUse() || job.state !== "succeeded") return;
      const transferKey = `${groupId}/${job.id}`;
      if (state.transfers.has(transferKey)) return;
      const credential = token(); const authVersion = state.authVersion;
      state.transfers.add(transferKey); renderGroups();
      try {
        const { response, body: blob } = await request(`/v1/audio/jobs/${encodeURIComponent(groupId)}/${encodeURIComponent(job.id)}/audio`, {}, "audio");
        if (!isCurrent(credential, authVersion)) return;
        const match = (response.headers.get("Content-Disposition") || "").match(/filename="([A-Za-z0-9._-]+\.wav)"/i);
        const filename = match ? match[1] : job.filename || "speech.wav";
        if (play) {
          clearOutput(); state.outputUrl = URL.createObjectURL(blob); state.outputJob = { groupId, job };
          outputAudio.src = state.outputUrl; outputAudio.hidden = false;
          outputLabel.textContent = `${filename} · ${job.voice_name || "Tanlangan ovoz"}`; outputLabel.hidden = false;
          outputAudio.setAttribute("aria-label", outputLabel.textContent);
          download.setAttribute("aria-label", `${filename} faylini yuklab olish`); download.hidden = false;
        }
        if (save) saveBlob(blob, filename);
        notify(`${filename} tayyor. Uni tinglashingiz yoki yuklab olishingiz mumkin.`, "success");
      } catch (error) { if (!error.stale) notify(errorMessage(error, "speech"), "error"); }
      finally { if (isCurrent(credential, authVersion)) { state.transfers.delete(transferKey); renderGroups(); } }
    }

    function renderGroups() {
      queuePanel.hidden = !canUse() || (!state.groups.length && !state.historyMore);
      historyButton.hidden = !state.historyMore;
      historyButton.disabled = !canUse() || state.historyLoading;
      historyButton.textContent = state.historyLoading ? "Oldingi natijalar yuklanmoqda…" : "Oldingi natijalar";
      const total = { queued: 0, running: 0, succeeded: 0, failed: 0 };
      const visibleGroups = new Set();
      const readySingles = [];
      for (const group of state.groups) {
        visibleGroups.add(group.id);
        const counts = countJobs(group);
        for (const name of Object.keys(total)) total[name] += counts[name];
        let view = state.groupViews.get(group.id);
        if (!view) {
          view = { title: element("h3"), list: element("ol", { className: "voice-studio__job-list" }), jobs: new Map() };
          view.root = element("section", { className: "voice-studio__job-group" }, view.title, view.list);
          state.groupViews.set(group.id, view);
        }
        view.title.textContent = `${group.jobs.length} ta matn · Tayyor: ${counts.succeeded} · Xatolik: ${counts.failed}`;
        for (const [index, job] of group.jobs.entries()) {
          let row = view.jobs.get(job.id);
          if (!row) {
            row = { name: element("strong"), text: element("p", { className: "voice-studio__job-text" }),
              voice: element("small"), status: element("span", { className: "voice-studio__job-status" }),
              play: element("button", { type: "button", className: "voice-studio__button voice-studio__button--soft", text: "Tinglash" }),
              download: element("button", { type: "button", className: "voice-studio__button voice-studio__button--outline", text: "Yuklab olish" }) };
            row.root = element("li", { className: "voice-studio__job" },
              element("div", { className: "voice-studio__job-content" }, row.name, row.text, row.voice),
              row.status, element("div", { className: "voice-studio__job-actions" }, row.play, row.download));
            row.play.addEventListener("click", () => transferJob(group.id, row.job, true, false));
            row.download.addEventListener("click", () => transferJob(group.id, row.job, false, true));
            view.jobs.set(job.id, row); view.list.append(row.root);
          }
          row.job = job; row.root.dataset.state = job.state; row.root.dataset.jobId = job.id;
          row.name.textContent = `${index + 1}. ${job.filename || `${job.key || "speech"}.wav`}`;
          row.text.textContent = job.text; row.text.title = job.text;
          row.voice.textContent = job.voice_name || "Tanlangan ovoz";
          row.status.textContent = { queued: "Navbatda", running: "Yaratilmoqda", succeeded: "Tayyor", failed: "Xatolik: audio yaratilmadi" }[job.state];
          row.play.hidden = row.download.hidden = job.state !== "succeeded";
          row.play.disabled = row.download.disabled = state.transfers.has(`${group.id}/${job.id}`) || !canUse();
          if (state.previewGroups.get(group.id) === job.id && ["succeeded", "failed"].includes(job.state)) {
            state.previewGroups.delete(group.id);
            if (job.state === "succeeded") readySingles.push({ groupId: group.id, job });
            else notify("Audio yaratilmadi. Matnni tekshirib qayta yuboring.", "error");
          }
        }
      }
      for (const id of state.groupViews.keys()) if (!visibleGroups.has(id)) state.groupViews.delete(id);
      const order = JSON.stringify(state.groups.map((group) => group.id));
      if (order !== state.groupOrder) {
        state.groupOrder = order; queueList.replaceChildren(...state.groups.map((group) => state.groupViews.get(group.id).root));
      }
      const execution = state.execution || total;
      const parallel = state.execution && Number.isInteger(state.execution.parallel_workers) ? `Bir vaqtda: ${state.execution.parallel_workers} ta · ` : "";
      queueSummary.textContent = `${parallel}Jarayonda: ${Math.max(execution.running || 0, total.running)} · Navbatda: ${Math.max(execution.queued || 0, total.queued)} · Tayyor: ${total.succeeded} · Xatolik: ${total.failed}`;
      download.disabled = !canUse() || !state.outputJob || state.transfers.has(`${state.outputJob.groupId}/${state.outputJob.job.id}`);
      for (const { groupId, job } of readySingles) transferJob(groupId, job, true, false);
    }

    function schedulePoll() {
      if (state.pollTimer !== null) clearTimeout(state.pollTimer);
      state.pollTimer = null;
      if (!canUse()) return;
      const pending = state.groups.some(pendingGroup);
      state.pollTimer = setTimeout(() => { state.pollTimer = null; loadGroups(); }, pending ? 1000 : 5000);
    }

    async function loadGroups() {
      if (!canUse() || state.groupsLoading) return;
      const version = state.groupsVersion; const loadId = ++state.groupsLoadId;
      state.groupsLoading = true;
      try {
        const { body } = await request("/v1/audio/jobs", {}, "json");
        if (version !== state.groupsVersion || loadId !== state.groupsLoadId) return;
        if (!body || !Array.isArray(body.data) || !body.data.every(validGroup)) throw new Error("invalid jobs response");
        mergeGroups(body.data); readHistory(body, false); state.execution = body.execution || null; renderGroups();
        const listed = new Set(body.data.map((group) => group.id));
        const missing = state.groups.filter((group) => pendingGroup(group) && !listed.has(group.id));
        let next = 0;
        // Follow every accepted waiter even when it just completed outside the recent history page.
        await Promise.all(Array.from({ length: Math.min(4, missing.length) }, async () => {
          while (next < missing.length && version === state.groupsVersion && loadId === state.groupsLoadId && canUse()) {
            const group = missing[next++];
            try {
              const { body: current } = await request(`/v1/audio/jobs/${encodeURIComponent(group.id)}`, {}, "json");
              if (version !== state.groupsVersion || loadId !== state.groupsLoadId) return;
              if (!validGroup(current) || current.id !== group.id) throw new Error("invalid group response");
              mergeGroups([current]); renderGroups();
            } catch (error) {
              if (!error.stale && canUse()) queueSummary.textContent = "Ayrim matnlar holati qayta tekshirilmoqda…";
            }
          }
        }));
      } catch (error) {
        if (!error.stale && canUse() && version === state.groupsVersion) {
          queueSummary.textContent = "Holatni yangilashda uzilish. Qayta tekshirilmoqda…";
        }
      } finally {
        if (loadId === state.groupsLoadId) { state.groupsLoading = false; schedulePoll(); }
      }
    }

    async function loadHistory() {
      if (!canUse() || state.historyLoading || !state.historyMore) return;
      const credential = token(); const authVersion = state.authVersion;
      const loadId = ++state.historyLoadId; const offset = state.historyNextOffset;
      state.historyLoading = true; renderGroups();
      try {
        const { body } = await request(`/v1/audio/jobs?limit=20&offset=${offset}`, {}, "json");
        if (!isCurrent(credential, authVersion) || loadId !== state.historyLoadId) return;
        if (!body || !Array.isArray(body.data) || !body.data.every(validGroup)) throw new Error("invalid jobs response");
        mergeGroups(body.data); readHistory(body, true); renderGroups();
      } catch (error) {
        if (!error.stale) notify(error.status === 401 ? errorMessage(error) : "Oldingi natijalarni yuklab bo‘lmadi. Qayta urinib ko‘ring.", "error");
      } finally {
        if (isCurrent(credential, authVersion) && loadId === state.historyLoadId) {
          state.historyLoading = false; renderGroups();
        }
      }
    }

    async function request(path, options, bodyType) {
      const credential = token();
      if (!credential || credential === state.rejectedToken) { const error = new Error("unauthorized"); error.status = 401; throw error; }
      const version = state.authVersion;
      const controller = new AbortController();
      state.controllers.add(controller);
      const perform = async () => {
        const response = await fetch(path, { ...options, signal: controller.signal, cache: "no-store",
          headers: { ...(options && options.headers), Authorization: `Bearer ${credential}` } });
        if (!isCurrent(credential, version)) { const error = new Error("stale authorization"); error.stale = true; throw error; }
        if (!response.ok) {
          if (response.status === 401) {
            state.rejectedToken = credential;
            state.catalogVersion += 1;
            state.authVersion += 1;
            state.loading = false;
            clearSample(); clearOutput(); clearJobs(); populateVoices([], "");
            notify("Kirish muddati tugagan yoki kalit noto‘g‘ri. Authorize orqali qayta kiring.", "error");
          }
          const error = new Error("request failed"); error.status = response.status; throw error;
        }
        let body = null;
        if (bodyType === "audio") {
          if (!/^audio\/(?:x-)?wav(?:\s*;|$)/i.test(response.headers.get("Content-Type") || "")) throw new Error("invalid audio response");
          body = await response.blob();
          if (!body.size) throw new Error("empty audio response");
        } else if (bodyType === "json") {
          body = await response.json();
        }
        if (!isCurrent(credential, version)) { const error = new Error("stale authorization"); error.stale = true; throw error; }
        return { response, body };
      };
      try {
        return await perform();
      } catch (error) {
        if (!isCurrent(credential, version) && error.status !== 401) error.stale = true;
        throw error;
      } finally { state.controllers.delete(controller); controller.abort(); }
    }

    async function refresh(preferred, quiet, afterMutation) {
      if (state.busy && !afterMutation) { state.refreshPending = true; return false; }
      if (!canUse() || state.loading) return false;
      const version = ++state.catalogVersion;
      const credential = state.token;
      state.loading = true; updateControls();
      if (!quiet) notify("Ovozlar ro‘yxati yuklanmoqda…");
      try {
        const { body } = await request("/v1/voices", {}, "json");
        if (version !== state.catalogVersion || token() !== credential || state.token !== credential) return false;
        if (!Array.isArray(body.data)) throw new Error("invalid response");
        populateVoices(body.data, preferred || state.selected);
        if (!quiet) notify("Ovozlar ro‘yxati yangilandi.", "success");
        return true;
      } catch (error) {
        if (!error.stale && version === state.catalogVersion) notify(errorMessage(error), "error");
        return false;
      } finally {
        if (version === state.catalogVersion) state.loading = false;
        updateControls();
      }
    }

    function syncAuth() {
      const credential = token();
      if (credential === state.token) return;
      state.token = credential; state.rejectedToken = null; state.authVersion += 1; state.catalogVersion += 1; state.loading = false;
      clearSample(); clearOutput(); clearJobs(); populateVoices([], "");
      if (credential) { notify("Ovozlar ro‘yxati yuklanmoqda…"); refresh("", false); loadGroups(); }
      else notify("Boshlash uchun Authorize orqali kiring.");
      updateControls();
    }

    authButton.addEventListener("click", () => {
      const authorize = document.querySelector("#swagger-ui .auth-wrapper .authorize");
      if (authorize) authorize.click();
      else notify("Sahifadagi Authorize tugmasi yuklanishini kuting.", "error");
    });
    document.addEventListener("click", (event) => {
      const reauthorize = event.target instanceof Element && event.target.closest(".swagger-ui .auth-btn-wrapper .auth");
      setTimeout(() => {
        if (reauthorize && state.rejectedToken && token() === state.rejectedToken) {
          state.rejectedToken = null;
          state.token = null;
        }
        syncAuth();
      }, 0);
    });
    const authTimer = setInterval(syncAuth, 1000);
    window.addEventListener("beforeunload", () => { clearInterval(authTimer); clearSample(); clearOutput(); clearJobs(); });
    refreshButton.addEventListener("click", () => refresh(state.selected, false));
    historyButton.addEventListener("click", loadHistory);
    voices.addEventListener("change", () => {
      state.selected = voices.value;
      clearSample(); updateControls();
      if (state.selected) { activate("speak"); notify("Ovoz tanlandi. Matn yozishingiz mumkin.", "success"); }
    });
    file.addEventListener("change", () => { if (file.files && file.files.length) { activate("save"); notify("Yozuv tanlandi. Ovoz nomi va turini kiriting."); } });
    saveTab.addEventListener("click", () => activate("save"));
    speakTab.addEventListener("click", () => activate("speak"));
    tabs.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const next = saveForm.hidden ? "save" : "speak";
      activate(next); (next === "save" ? saveTab : speakTab).focus();
    });
    speechText.addEventListener("input", () => { counter.textContent = `${speechText.value.length} belgi`; });
    key.addEventListener("input", updateInputMode);
    inputMode.addEventListener("change", updateInputMode);
    bulkText.addEventListener("input", () => {
      try { bulkCounter.textContent = `${bulkItems(bulkText.value).length} ta matn`; }
      catch (error) { bulkCounter.textContent = error.message; }
    });
    bulkFile.addEventListener("change", async () => {
      const upload = bulkFile.files && bulkFile.files[0];
      if (!upload) return;
      try {
        const content = await upload.text(); const items = bulkItems(content);
        bulkText.value = content; bulkCounter.textContent = `${items.length} ta matn`;
        notify(`${items.length} ta matn ochildi. Yuborishdan oldin tekshiring.`, "success");
      } catch (error) { notify(error.message || "Faylni ochib bo‘lmadi.", "error"); }
    });
    download.addEventListener("click", (event) => {
      event.preventDefault();
      if (state.outputJob) transferJob(state.outputJob.groupId, state.outputJob.job, false, true);
    });

    saveForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (state.busy || state.loading) return;
      const upload = file.files && file.files[0];
      if (!upload) { notify("Avval yangi ovoz yozuvini tanlang.", "error"); file.focus(); return; }
      if (!name.value.trim() || !gender.value) { notify("Ovoz nomi va turini kiriting.", "error"); return; }
      const credential = token(); const authVersion = state.authVersion;
      if (!isCurrent(credential, authVersion)) { syncAuth(); notify("Authorize orqali qayta kiring.", "error"); return; }
      const form = new FormData();
      form.append("ref_audio", upload);
      form.append("name", name.value.trim());
      form.append("gender", gender.value);
      form.append("reference_language", referenceLanguage.value);
      if (transcript.value.trim()) form.append("ref_text", transcript.value.trim());
      state.catalogVersion += 1;
      setBusy(true); notify("Ovoz saqlanmoqda. Biroz kuting…");
      let focusText = false;
      try {
        const { body: voice } = await request("/v1/voices/clone", { method: "POST", body: form }, "json");
        if (!isCurrent(credential, authVersion)) { syncAuth(); return; }
        if (!voice || typeof voice.id !== "string" || typeof voice.name !== "string") throw new Error("invalid response");
        clearSample();
        state.voices = state.voices.filter((item) => item.id !== voice.id).concat([voice]);
        populateVoices(state.voices, voice.id);
        activate("speak");
        state.catalogVersion += 1;
        const loaded = await refresh(voice.id, true, true);
        if (state.selected === voice.id && isCurrent(credential, authVersion)) {
          const preparation = voice.reference_preparation;
          const excerptSelected = preparation && Number.isFinite(preparation.source_seconds) &&
            Number.isFinite(preparation.selected_seconds) && preparation.selected_seconds > 0 &&
            preparation.selected_seconds < preparation.source_seconds;
          const excerptNotice = excerptSelected ? "Yozuvdan ovozli namuna tanlandi. " +
            (preparation.automatic_transcript ? "Namuna matni avtomatik aniqlanib, ishlatildi. " : "") : "";
          notify("Ovoz saqlandi. " + excerptNotice + (loaded ? "Endi matn yozing va ovoz yarating." :
            "Matn yozishingiz mumkin; ro‘yxatni keyin yangilang."), "success");
          focusText = true;
        }
      } catch (error) { if (!error.stale) notify(errorMessage(error, "save"), "error"); }
      finally { setBusy(false); if (focusText && isCurrent(credential, authVersion)) speechText.focus(); }
    });

    speakForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (state.busy || state.loading) return;
      const selected = state.selected;
      const text = speechText.value.trim();
      const downloadKey = key.value.trim();
      const bulk = inputMode.value === "bulk";
      const credential = token(); const authVersion = state.authVersion;
      if (!isCurrent(credential, authVersion)) { syncAuth(); notify("Authorize orqali qayta kiring.", "error"); return; }
      if (!selected || !currentVoice()) { notify("Avval ovozni tanlang.", "error"); voices.focus(); return; }
      if (!bulk && !text) { notify("Matnni kiriting.", "error"); speechText.focus(); return; }
      if (downloadKey && !validKey(downloadKey)) {
        notify("Fayl nomida faqat lotin harflari, raqam, nuqta, _ va - ishlating.", "error"); key.focus(); return;
      }
      let items;
      try {
        items = bulk ? bulkItems(bulkText.value) : [{ text, key: downloadKey }];
        if (!items.length) throw new Error("Matnlarni kiriting.");
        const usedKeys = new Set();
        items = items.map((item, index) => {
          const itemKey = item.key || (bulk ? `${downloadKey || "audio"}.${String(index + 1).padStart(3, "0")}` : "");
          if (itemKey && !validKey(itemKey)) throw new Error("Fayl nomi juda uzun. Qisqaroq nom kiriting.");
          if (itemKey && usedKeys.has(itemKey.toLowerCase())) throw new Error("Har bir matnning fayl nomi alohida bo‘lsin.");
          usedKeys.add(itemKey.toLowerCase());
          const payload = { voice: selected, text: item.text, language: speechLanguage.value };
          if (itemKey) payload.key = itemKey;
          return payload;
        });
      } catch (error) { notify(error.message, "error"); (bulk ? bulkText : speechText).focus(); return; }
      setBusy(true); notify(`${items.length} ta matn navbatga yuborilmoqda…`);
      try {
        const { body: group } = await request("/v1/audio/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ items }) }, "json");
        if (!isCurrent(credential, authVersion)) { syncAuth(); return; }
        if (!validGroup(group) || group.jobs.length !== items.length) throw new Error("invalid jobs response");
        state.groupsVersion += 1; state.groupsLoadId += 1; state.groupsLoading = false;
        state.groups = [group, ...state.groups.filter((item) => item.id !== group.id)];
        if (!bulk && group.jobs.length === 1) state.previewGroups.set(group.id, group.jobs[0].id);
        notify(`${items.length} ta matn qabul qilindi. Boshqa matnlarni ham yuborishingiz mumkin.`, "success");
        renderGroups(); schedulePoll();
      } catch (error) {
        if (!error.stale) {
          notify(errorMessage(error, "enqueue"), "error");
          if (!error.status || error.status === 504) loadGroups();
        }
      }
      finally { setBusy(false); }
    });

    sampleButton.addEventListener("click", async () => {
      if (state.busy || state.loading || !state.selected) return;
      const selected = state.selected;
      const credential = token(); const authVersion = state.authVersion;
      if (!isCurrent(credential, authVersion)) { syncAuth(); notify("Authorize orqali qayta kiring.", "error"); return; }
      clearSample(); const version = state.sampleVersion;
      setBusy(true); notify("Ovoz namunasi yuklanmoqda…");
      try {
        const { body: blob } = await request(`/v1/voices/${encodeURIComponent(selected)}/sample`, {}, "audio");
        if (!isCurrent(credential, authVersion) || state.selected !== selected || version !== state.sampleVersion) { syncAuth(); return; }
        state.sampleUrl = URL.createObjectURL(blob);
        sampleAudio.src = state.sampleUrl; sampleAudio.hidden = false;
        notify("Ovoz namunasi tayyor. Tinglash tugmasini bosing.", "success");
        sampleAudio.focus();
      } catch (error) { if (!error.stale) notify(errorMessage(error, "sample"), "error"); }
      finally { setBusy(false); }
    });

    deleteButton.addEventListener("click", async () => {
      const voice = currentVoice();
      if (state.busy || state.loading || !voice || !["clone", "design"].includes(voice.source)) return;
      const credential = token(); const authVersion = state.authVersion;
      if (!isCurrent(credential, authVersion)) { syncAuth(); notify("Authorize orqali qayta kiring.", "error"); return; }
      if (!window.confirm(`“${voice.name}” ovozini o‘chirmoqchimisiz?`)) return;
      const selected = voice.id;
      state.catalogVersion += 1;
      setBusy(true); notify("Ovoz o‘chirilmoqda…");
      try {
        await request(`/v1/voices/${encodeURIComponent(selected)}`, { method: "DELETE" }, "json");
        if (!isCurrent(credential, authVersion) || state.selected !== selected) { syncAuth(); return; }
        state.voices = state.voices.filter((item) => item.id !== selected);
        state.selected = ""; clearSample(); clearOutput(); populateVoices(state.voices, "");
        state.catalogVersion += 1;
        await refresh("", true, true);
        if (isCurrent(credential, authVersion)) notify("Ovoz o‘chirildi.", "success");
      } catch (error) { if (!error.stale) notify(errorMessage(error, "delete"), "error"); }
      finally { setBusy(false); }
    });

    populateVoices([], "");
    syncAuth();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount, { once: true });
  else mount();
}());
