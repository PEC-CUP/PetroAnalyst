const pretty = value => JSON.stringify(value, null, 2);

async function requestJson(url, payload) {
  const response = await fetch(url, payload ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  } : {});
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = {status: "error", message: text || response.statusText};
  }
  if (!response.ok) throw new Error(pretty(data));
  return data;
}

function byId(id) {
  return document.getElementById(id);
}

function optionText(item) {
  return `${item.module} / ${item.key}${item.overridden ? " *" : ""}`;
}

function renderPromptSelect(prompts, selectedKey) {
  return prompts.map(item => {
    const selected = item.key === selectedKey ? " selected" : "";
    return `<option value="${item.key}"${selected}>${optionText(item)}</option>`;
  }).join("");
}

function setPromptEditor(state) {
  const selected = state.prompts.find(item => item.key === byId("llmPromptKey").value) || state.prompts[0];
  if (!selected) return;
  byId("llmPromptKey").value = selected.key;
  byId("llmPromptDescription").textContent = `${selected.module}: ${selected.description}${selected.overridden ? " (overridden)" : " (default)"}`;
  byId("llmPromptTemplate").value = selected.template;
}

function fillConfig(config) {
  byId("llmDeepseekBaseUrl").value = config.deepseek.base_url || "";
  byId("llmDeepseekTrustEnv").checked = Boolean(config.deepseek.trust_env);
  byId("llmDeepseekChatModel").value = config.deepseek.chat_model || "";
  byId("llmDeepseekReasonerModel").value = config.deepseek.reasoner_model || "";
  byId("llmOpenaiBaseUrl").value = config.openai_expert.base_url || "";
  byId("llmOpenaiModel").value = config.openai_expert.model || "";
  byId("llmOpenaiStrongModel").value = config.openai_expert.strong_model || "";
  byId("llmTimeout").value = config.timeout_seconds || "";
  byId("llmConfigSummary").textContent = [
    `DeepSeek key: ${config.deepseek.api_key_configured ? "configured" : "missing"}`,
    `OpenAI key: ${config.openai_expert.api_key_configured ? "configured" : "missing"}`,
    `DeepSeek model: ${config.deepseek.chat_model || "-"}`,
    `OpenAI expert model: ${config.openai_expert.strong_model || config.openai_expert.model || "-"}`
  ].join(" | ");
}

function panelHtml(title) {
  return `
    <section class="card">
      <h2>${title || "LLM API and Prompt Configuration"}</h2>
      <p class="action-guide">Review and adjust runtime LLM providers, model versions, keys, and module prompts. Key fields are write-only: leave them blank to keep the current key.</p>
      <div class="grid">
        <div>
          <label>DeepSeek base URL</label>
          <input id="llmDeepseekBaseUrl" placeholder="https://api.deepseek.com">
        </div>
        <div>
          <label>DeepSeek API key</label>
          <input id="llmDeepseekKey" type="password" autocomplete="off" placeholder="leave blank to keep current key">
        </div>
        <div>
          <label>DeepSeek chat model</label>
          <input id="llmDeepseekChatModel" placeholder="deepseek-chat">
        </div>
        <div>
          <label>DeepSeek reasoner model</label>
          <input id="llmDeepseekReasonerModel" placeholder="deepseek-reasoner">
        </div>
        <div>
          <label>OpenAI expert base URL</label>
          <input id="llmOpenaiBaseUrl" placeholder="https://api.openai.com/v1">
        </div>
        <div>
          <label>OpenAI expert API key</label>
          <input id="llmOpenaiKey" type="password" autocomplete="off" placeholder="leave blank to keep current key">
        </div>
        <div>
          <label>OpenAI default model</label>
          <input id="llmOpenaiModel" placeholder="gpt-5.4-mini">
        </div>
        <div>
          <label>OpenAI strong model</label>
          <input id="llmOpenaiStrongModel" placeholder="gpt-5.5">
        </div>
        <div>
          <label>LLM timeout seconds</label>
          <input id="llmTimeout" type="number" min="1" max="600">
        </div>
        <label class="check"><input id="llmDeepseekTrustEnv" type="checkbox"> DeepSeek uses system proxy variables</label>
      </div>
      <div class="compare-actions">
        <button class="teal" type="button" id="llmSaveConfigBtn">Save LLM config</button>
        <button type="button" id="llmTestDeepseekBtn">Test DeepSeek</button>
        <button type="button" id="llmTestOpenaiBtn">Test OpenAI expert</button>
        <button type="button" id="llmRefreshBtn">Refresh</button>
      </div>
      <div class="note" id="llmConfigSummary">Loading LLM configuration...</div>
      <div class="grid" style="margin-top:14px">
        <div>
          <label>Prompt template</label>
          <select id="llmPromptKey"></select>
          <p class="field-help" id="llmPromptDescription"></p>
        </div>
        <div>
          <label>Prompt test text</label>
          <input id="llmTestPrompt" value="Reply with OK only.">
        </div>
      </div>
      <textarea id="llmPromptTemplate" style="min-height:220px"></textarea>
      <div class="compare-actions">
        <button class="teal" type="button" id="llmSavePromptBtn">Save prompt</button>
        <button type="button" id="llmResetPromptBtn">Reset selected prompt</button>
      </div>
      <pre id="llmOut">{}</pre>
    </section>
  `;
}

export function mountLlmConsole(containerId = "llmConfigPanel", options = {}) {
  const root = byId(containerId);
  if (!root) return;
  root.innerHTML = panelHtml(options.title);
  const state = {prompts: []};

  async function refresh() {
    try {
      const data = await requestJson("/support-layer/llm/status");
      state.prompts = data.prompts || [];
      fillConfig(data.config);
      const selectedKey = byId("llmPromptKey").value || options.promptKey || "";
      byId("llmPromptKey").innerHTML = renderPromptSelect(state.prompts, selectedKey);
      setPromptEditor(state);
      byId("llmOut").textContent = pretty(data.config);
    } catch (error) {
      byId("llmOut").textContent = String(error);
    }
  }

  async function saveConfig() {
    const deepseek = {
      base_url: byId("llmDeepseekBaseUrl").value,
      trust_env: byId("llmDeepseekTrustEnv").checked,
      chat_model: byId("llmDeepseekChatModel").value,
      reasoner_model: byId("llmDeepseekReasonerModel").value
    };
    const deepseekKey = byId("llmDeepseekKey").value.trim();
    if (deepseekKey) deepseek.api_key = deepseekKey;
    const openaiExpert = {
      base_url: byId("llmOpenaiBaseUrl").value,
      model: byId("llmOpenaiModel").value,
      strong_model: byId("llmOpenaiStrongModel").value
    };
    const openaiKey = byId("llmOpenaiKey").value.trim();
    if (openaiKey) openaiExpert.api_key = openaiKey;
    const payload = {
      deepseek,
      openai_expert: openaiExpert,
      timeout_seconds: Number(byId("llmTimeout").value) || null
    };
    const data = await requestJson("/support-layer/llm/config", payload);
    byId("llmDeepseekKey").value = "";
    byId("llmOpenaiKey").value = "";
    fillConfig(data.config);
    byId("llmOut").textContent = pretty(data);
  }

  async function savePrompt(reset = false) {
    const data = await requestJson("/support-layer/llm/prompts", {
      key: byId("llmPromptKey").value,
      template: byId("llmPromptTemplate").value,
      reset
    });
    state.prompts = data.prompts || [];
    byId("llmPromptKey").innerHTML = renderPromptSelect(state.prompts, data.prompt.key);
    setPromptEditor(state);
    byId("llmOut").textContent = pretty(data.prompt);
  }

  async function test(provider) {
    const model = provider === "openai" ? byId("llmOpenaiModel").value : byId("llmDeepseekChatModel").value;
    const data = await requestJson("/support-layer/llm/test", {
      provider,
      model,
      prompt: byId("llmTestPrompt").value
    });
    byId("llmOut").textContent = pretty(data);
  }

  byId("llmPromptKey").addEventListener("change", () => setPromptEditor(state));
  byId("llmRefreshBtn").addEventListener("click", refresh);
  byId("llmSaveConfigBtn").addEventListener("click", () => saveConfig().catch(error => { byId("llmOut").textContent = String(error); }));
  byId("llmSavePromptBtn").addEventListener("click", () => savePrompt(false).catch(error => { byId("llmOut").textContent = String(error); }));
  byId("llmResetPromptBtn").addEventListener("click", () => savePrompt(true).catch(error => { byId("llmOut").textContent = String(error); }));
  byId("llmTestDeepseekBtn").addEventListener("click", () => test("deepseek").catch(error => { byId("llmOut").textContent = String(error); }));
  byId("llmTestOpenaiBtn").addEventListener("click", () => test("openai").catch(error => { byId("llmOut").textContent = String(error); }));
  refresh();
}

