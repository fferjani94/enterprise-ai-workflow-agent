(() => {
  'use strict';

  const API = '/api/v1/investigations';
  const FALLBACK_CONFIG = {
    model: 'unknown', max_rounds: 4, max_rounds_limit: 8,
    max_question_length: 4000, timeout_seconds: 300,
  };

  // Titles for the structured error codes returned by the backend.
  const ERROR_TITLES = {
    invalid_request: 'Invalid input',
    ollama_unavailable: 'Ollama is not reachable',
    ollama_integration_missing: 'Ollama integration is not installed',
    model_unavailable: 'Model not installed',
    model_inference_failed: 'The model failed while investigating',
    mcp_unavailable: 'Tool server unavailable',
    timeout: 'Investigation timed out',
    internal_error: 'Unexpected server error',
  };

  const EVIDENCE_GROUPS = [
    ['account', 'Accounts'],
    ['ticket', 'Support tickets'],
    ['product', 'Products'],
    ['document', 'Knowledge documents'],
  ];

  const REF_NAMES = {
    account: 'Account', ticket: 'Ticket', product: 'Product', document: 'Document',
  };

  // How retrieved records relate to the question, as labelled by the backend.
  const RELEVANCE = {
    relevant: ['Matches question topic', 'badge-ok'],
    no_match: ['No topic match', 'badge-warn'],
    context: ['Context', 'badge-muted'],
    unassessed: ['Relevance not assessed', 'badge-muted'],
  };

  const ANALYSIS_SECTIONS = [
    ['findings', 'Relevant support tickets'],
    ['guidance', 'Relevant knowledge documents (general guidance, not a diagnosis)'],
    ['context', 'Other retrieved records: context'],
    ['unsupported', 'Other retrieved records: not evidence of this issue'],
    ['unconfirmed', 'Unknown or unconfirmed'],
  ];

  const LABEL_WORDS = { id: 'ID', url: 'URL' };
  const DEFAULT_LIMITED_TEXT = 'The agent did not retrieve all the evidence this question needs, '
    + 'so there is no complete conclusion. The evidence and timeline below show what was '
    + 'retrieved. Try a narrower question or allow more tool rounds.';
  const ANSWER_SOURCES = {
    model: 'Written by the local model. Check it against the findings and records below.',
    synthesis: 'Written by the local model from the verified records only, without tools. '
      + 'Check it against the findings and records below.',
    evidence_summary: 'Assembled from the verified records. It contains no model-generated '
      + 'conclusion.',
  };

  const $ = (id) => document.getElementById(id);
  const ui = {
    form: $('investigation-form'), question: $('question'), questionError: $('question-error'),
    charCount: $('char-count'), rounds: $('max-rounds'), submit: $('submit-btn'),
    stop: $('stop-btn'), stopHint: $('stop-hint'), announcer: $('status-announcer'),
    empty: $('state-empty'), loading: $('state-loading'), notice: $('state-notice'),
    error: $('state-error'), result: $('state-result'), elapsed: $('elapsed'),
  };
  const state = { config: FALLBACK_CONFIG, controller: null, timer: null };

  // Builds DOM nodes with textContent only; strings are never parsed as markup.
  function h(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === 'class') node.className = value;
      else node.setAttribute(key, value === true ? '' : String(value));
    }
    for (const child of children.flat()) {
      if (child != null) node.append(child);
    }
    return node;
  }

  const clear = (node) => node.replaceChildren();
  const announce = (message) => { ui.announcer.textContent = message; };

  /* ---------- Configuration ---------- */

  function validConfig(c) {
    return c && typeof c.model === 'string' && Number.isInteger(c.max_rounds)
      && Number.isInteger(c.max_rounds_limit) && Number.isInteger(c.max_question_length)
      && typeof c.timeout_seconds === 'number';
  }

  function setRuntime(status, ok) {
    const el = $('rt-status');
    el.textContent = status;
    el.classList.toggle('is-ok', ok);
    el.classList.toggle('is-bad', !ok);
  }

  function applyConfig(config, online) {
    state.config = config;
    $('model-badge').textContent = `Model: ${online ? config.model : 'unavailable'}`;
    $('rt-model').textContent = online ? config.model : '-';
    $('rt-rounds').textContent = online ? String(config.max_rounds_limit) : '-';
    $('rt-timeout').textContent = online ? `${config.timeout_seconds} s` : '-';
    ui.question.maxLength = config.max_question_length;
    const previous = Number(ui.rounds.value);
    clear(ui.rounds);
    for (let n = 1; n <= config.max_rounds_limit; n += 1) {
      ui.rounds.append(h('option', { value: n }, String(n)));
    }
    const keep = previous >= 1 && previous <= config.max_rounds_limit;
    ui.rounds.value = String(keep ? previous : config.max_rounds);
    updateCounter();
  }

  async function loadConfig() {
    try {
      const response = await fetch(`${API}/config`, { headers: { Accept: 'application/json' } });
      const config = response.ok ? await response.json() : null;
      if (!validConfig(config)) throw new Error('bad config');
      applyConfig(config, true);
      setRuntime('Online', true);
    } catch {
      applyConfig(FALLBACK_CONFIG, false);
      setRuntime('Unreachable', false);
    }
  }

  /* ---------- Form ---------- */

  function updateCounter() {
    const length = ui.question.value.length;
    const max = state.config.max_question_length;
    ui.charCount.textContent = `${length} / ${max}`;
    ui.charCount.classList.toggle('char-over', length > max);
  }

  function setFieldError(message) {
    ui.questionError.textContent = message;
    ui.question.setAttribute('aria-invalid', message ? 'true' : 'false');
  }

  function setBusy(busy) {
    ui.submit.disabled = busy;
    ui.submit.textContent = busy ? 'Investigating...' : 'Investigate';
    ui.rounds.disabled = busy;
    ui.question.readOnly = busy;
    ui.stop.hidden = !busy;
    ui.stopHint.hidden = !busy;
    ui.form.setAttribute('aria-busy', String(busy));
    document.querySelectorAll('.chip').forEach((chip) => { chip.disabled = busy; });
  }

  /* ---------- Result panels ---------- */

  function showOnly(panel) {
    for (const el of [ui.empty, ui.loading, ui.notice, ui.error, ui.result]) {
      el.hidden = el !== panel;
    }
  }

  function startTimer() {
    const started = performance.now();
    const tick = () => {
      const seconds = (performance.now() - started) / 1000;
      ui.elapsed.textContent = seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds)}s`;
    };
    tick();
    state.timer = setInterval(tick, 200);
  }

  function stopTimer() {
    clearInterval(state.timer);
    state.timer = null;
  }

  function showLoading() {
    $('loading-detail').textContent =
      'The agent is calling read-only tools and the local model. Small local models can take ' +
      `a minute or more. The server stops after ${state.config.timeout_seconds} seconds.`;
    showOnly(ui.loading);
    startTimer();
    announce('Investigation started.');
  }

  function showError({ title, message, details, reference }) {
    $('error-title').textContent = title;
    $('error-message').textContent = message;
    const list = $('error-details');
    clear(list);
    (details || []).forEach((item) => list.append(h('li', {}, item)));
    list.hidden = !(details && details.length);
    const ref = $('error-ref');
    ref.textContent = reference ? `Reference: ${reference}` : '';
    ref.hidden = !reference;
    showOnly(ui.error);
    announce(`Investigation failed: ${title}.`);
  }

  function errorFromResponse(status, body) {
    const error = body && typeof body.error === 'object' && body.error ? body.error : null;
    if (!error || typeof error.code !== 'string') {
      return {
        title: 'Unexpected response',
        message: `The server returned HTTP ${status} in an unexpected format. Check the server logs.`,
      };
    }
    const details = Array.isArray(error.details)
      ? error.details.map((d) => `${d.field}: ${d.message}`)
      : [];
    return {
      title: ERROR_TITLES[error.code] || 'Investigation failed',
      message: String(error.message || 'The investigation failed.'),
      details,
      reference: typeof body.id === 'string' ? body.id : '',
    };
  }

  /* ---------- Rendering ---------- */

  function humanLabel(key) {
    return key.split('_')
      .map((word, i) => LABEL_WORDS[word] || (i === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word))
      .join(' ');
  }

  function parseUrl(raw) {
    try { return new URL(raw); } catch { return null; }
  }

  // Reserved or documentation-only hosts: the seeded knowledge URLs are not real pages.
  function isSyntheticHost(host) {
    return /(^|\.)(invalid|example|test|localhost)$/.test(host)
      || /(^|\.)example\.(com|org|net)$/.test(host);
  }

  // Only web URLs become links, and synthetic ones are shown as plain text with a note.
  function linkNode(label, raw) {
    const url = parseUrl(raw);
    if (!url || (url.protocol !== 'https:' && url.protocol !== 'http:')) {
      return label === raw ? raw : `${label} (${raw})`;
    }
    if (isSyntheticHost(url.hostname)) {
      return h('span', { class: 'synthetic-url' },
        h('code', {}, label),
        h('span', { class: 'badge badge-muted' }, 'Example URL, not a live link'));
    }
    return h('a', { href: url.href, target: '_blank', rel: 'noopener noreferrer nofollow' }, label);
  }

  /* Minimal Markdown (bold, italics, code, links, lists, headings, code blocks).
     DOM nodes are built directly, so model output is never parsed as HTML. */

  const INLINE = /\*\*([^*]+?)\*\*|`([^`]+)`|\*([^*\s][^*]*?)\*|\[([^\]]+)\]\(([^)\s]+)\)/;
  const LIST_ITEM = /^(\s*)([-*\u2022]|\d+[.)])\s+(.*)$/;

  function inline(text) {
    const nodes = [];
    let rest = text;
    while (rest) {
      const match = INLINE.exec(rest);
      if (!match) { nodes.push(rest); break; }
      if (match.index) nodes.push(rest.slice(0, match.index));
      if (match[1] !== undefined) nodes.push(h('strong', {}, inline(match[1])));
      else if (match[2] !== undefined) nodes.push(h('code', {}, match[2]));
      else if (match[3] !== undefined) nodes.push(h('em', {}, inline(match[3])));
      else nodes.push(linkNode(match[4], match[5]));
      rest = rest.slice(match.index + match[0].length);
    }
    return nodes;
  }

  function renderMarkdown(source) {
    const root = document.createDocumentFragment();
    let paragraph = [];
    let lists = [];
    let code = null;

    const flushParagraph = () => {
      if (!paragraph.length) return;
      const node = h('p', {});
      paragraph.forEach((line, i) => {
        if (i) node.append(h('br'));
        node.append(...inline(line));
      });
      root.append(node);
      paragraph = [];
    };
    const codeBlock = (lines) => h('pre', { class: 'code-block' }, h('code', {}, lines.join('\n')));

    for (const line of source.replace(/\r\n?/g, '\n').split('\n')) {
      if (code) {
        if (/^\s*```/.test(line)) { root.append(codeBlock(code)); code = null; } else code.push(line);
        continue;
      }
      if (/^\s*```/.test(line)) { flushParagraph(); lists = []; code = []; continue; }
      if (!line.trim()) { flushParagraph(); lists = []; continue; }
      const heading = /^\s{0,3}#{1,6}\s+(.*)$/.exec(line);
      if (heading) {
        flushParagraph(); lists = [];
        root.append(h('h4', { class: 'md-heading' }, inline(heading[1])));
        continue;
      }
      const item = LIST_ITEM.exec(line);
      if (item) {
        flushParagraph();
        const indent = item[1].replace(/\t/g, '    ').length;
        const tag = /\d/.test(item[2]) ? 'OL' : 'UL';
        while (lists.length && indent < lists[lists.length - 1].indent) lists.pop();
        // A list of the other kind at the same indentation starts a new list.
        if (lists.length && indent === lists[lists.length - 1].indent
            && lists[lists.length - 1].el.tagName !== tag) lists.pop();
        let top = lists[lists.length - 1];
        if (!top || indent > top.indent) {
          const el = h(tag.toLowerCase(), { class: 'md-list' });
          ((top && top.el.lastElementChild) || root).append(el);
          top = { indent, el };
          lists.push(top);
        }
        top.el.append(h('li', {}, inline(item[3])));
        continue;
      }
      lists = [];
      paragraph.push(line.trim());
    }
    flushParagraph();
    if (code) root.append(codeBlock(code));
    return root;
  }

  function formatValue(key, value) {
    if (typeof value === 'string' && /(^|_)url$/.test(key)) return linkNode(value, value);
    return typeof value === 'object' ? JSON.stringify(value) : String(value);
  }

  function renderRecord(record) {
    const hasTitle = typeof record.title === 'string' && record.title !== '';
    const fields = Object.entries(record.fields || {}).filter(([, v]) => v != null && v !== '');
    const relevance = RELEVANCE[record.relevance];
    const unmatched = record.relevance === 'no_match';
    return h('article', { class: unmatched ? 'record is-unmatched' : 'record' },
      h('div', { class: 'record-head' },
        h('span', { class: 'record-id' }, `#${record.id}`),
        h('span', { class: hasTitle ? 'record-title' : 'record-title is-missing' },
          hasTitle ? record.title : 'ID only (details not retrieved)')),
      relevance ? h('span', { class: `badge ${relevance[1]}` }, relevance[0]) : null,
      unmatched
        ? h('p', { class: 'record-note' },
          'Nothing in the retrieved fields of this record matches the question. ' +
          'It is not evidence of this issue.')
        : null,
      record.description ? h('p', { class: 'record-desc' }, record.description) : null,
      record.relevance === 'relevant' && typeof record.relevance_basis === 'string'
        ? h('p', { class: 'record-note' }, record.relevance_basis) : null,
      fields.length
        ? h('dl', { class: 'record-fields' },
          fields.map(([key, value]) => h('div', {},
            h('dt', {}, humanLabel(key)), h('dd', {}, formatValue(key, value)))))
        : null);
  }

  function renderAnalysis(analysis) {
    const box = $('analysis');
    clear(box);
    let any = false;
    for (const [key, title] of ANALYSIS_SECTIONS) {
      const items = analysis && Array.isArray(analysis[key]) ? analysis[key] : [];
      if (!items.length) continue;
      any = true;
      box.append(h('section', { class: `analysis-group analysis-${key}` },
        h('h4', {}, title),
        h('ul', { class: 'analysis-list' }, items.map((item) => h('li', {},
          h('span', {}, item.text),
          Array.isArray(item.evidence) && item.evidence.length
            ? h('span', { class: 'refs' }, item.evidence.slice(0, 6).map((ref) =>
              h('span', { class: 'ref' }, `${REF_NAMES[ref.type] || ref.type} #${ref.id}`)))
            : null)))));
    }
    if (!any) {
      box.append(h('p', { class: 'muted' },
        'No records were retrieved, so there are no findings to report.'));
    }
  }

  function renderEvidence(evidence) {
    const container = $('evidence');
    clear(container);
    $('evidence-count').textContent = `${evidence.length} record${evidence.length === 1 ? '' : 's'}`;
    if (!evidence.length) {
      container.append(h('p', { class: 'muted' },
        'No records were retrieved, so this answer has no verified evidence. ' +
        'Review the tool timeline to see what the agent tried.'));
      return;
    }
    for (const [type, label] of EVIDENCE_GROUPS) {
      const records = evidence.filter((record) => record.type === type);
      if (!records.length) continue;
      container.append(h('section', { class: 'evidence-group' },
        h('h4', {}, `${label} (${records.length})`),
        h('div', { class: 'records' }, records.map(renderRecord))));
    }
  }

  function jsonBlock(title, value) {
    return h('div', {}, h('h5', {}, title),
      h('pre', { class: 'code-block' }, JSON.stringify(value, null, 2)));
  }

  function renderStep(entry, index) {
    const ok = entry.success === true;
    const auto = entry.origin === 'orchestrator';
    const body = h('div', { class: 'step-body' });
    if (auto) {
      body.append(h('div', {}, h('h5', {}, 'Added by the orchestrator'),
        h('p', { class: 'small' }, entry.reason
          || 'The model stopped before retrieving evidence the question asks for.')));
    }
    // The API serializes missing arguments as null (e.g. calls over the per-round limit).
    body.append(entry.arguments == null
      ? h('p', { class: 'muted small' }, 'No arguments were recorded for this call.')
      : jsonBlock('Arguments', entry.arguments));
    if (entry.normalized_arguments != null
        && JSON.stringify(entry.normalized_arguments) !== JSON.stringify(entry.arguments)) {
      body.append(jsonBlock('Arguments sent to the tool', entry.normalized_arguments));
    }
    if (Array.isArray(entry.adjustments) && entry.adjustments.length) {
      body.append(h('div', {}, h('h5', {}, 'Adjusted arguments'),
        h('ul', { class: 'small' }, entry.adjustments.map((note) => h('li', {}, note)))));
    }
    if (entry.error) {
      body.append(h('div', {}, h('h5', {}, 'Error'), h('p', { class: 'step-error' }, entry.error)));
    }
    return h('li', {},
      h('details', { class: `step-card ${ok ? 'ok' : 'fail'}` },
        h('summary', {},
          h('span', { class: 'step-index' }, String(index + 1)),
          h('span', { class: 'step-tool' }, entry.tool),
          h('span', { class: `badge ${ok ? 'badge-ok' : 'badge-fail'}` }, ok ? 'Success' : 'Failed'),
          auto ? h('span', { class: 'badge badge-muted', title: 'Added by the orchestrator' },
            'Auto') : null,
          ok && Number.isInteger(entry.record_count)
            ? h('span', { class: 'step-meta' },
              `${entry.record_count} record${entry.record_count === 1 ? '' : 's'}`)
            : null),
        body));
  }

  function renderTimeline(trace) {
    const list = $('timeline');
    clear(list);
    const failed = trace.filter((entry) => entry.success !== true).length;
    $('timeline-count').textContent = trace.length
      ? `${trace.length} call${trace.length === 1 ? '' : 's'}${failed ? `, ${failed} failed` : ''}`
      : 'no calls';
    $('toggle-steps').hidden = trace.length === 0;
    $('toggle-steps').textContent = 'Expand all steps';
    if (!trace.length) {
      list.append(h('li', { class: 'muted' }, 'The agent answered without calling any tools.'));
      return;
    }
    trace.forEach((entry, index) => list.append(renderStep(entry, index)));
  }

  function validResult(data) {
    return data && typeof data.answer === 'string' && typeof data.status === 'string'
      && Array.isArray(data.tool_trace) && Array.isArray(data.evidence);
  }

  function renderResult(data) {
    const limited = data.status === 'limited';
    const status = $('result-status');
    status.textContent = limited ? 'Limited' : 'Completed';
    status.className = `status ${limited ? 'status-limited' : 'status-completed'}`;
    $('limited-banner').hidden = !limited;
    $('limited-text').textContent =
      typeof data.limited_reason === 'string' && data.limited_reason
        ? data.limited_reason : DEFAULT_LIMITED_TEXT;
    $('result-question').textContent = data.question;
    $('m-duration').textContent = `${Number(data.duration_seconds).toFixed(1)} s`;
    $('m-rounds').textContent = String(data.tool_rounds);
    $('m-calls').textContent = String(data.tool_calls);
    $('m-evidence').textContent = String(data.evidence.length);
    $('answer').replaceChildren(
      renderMarkdown(typeof data.narrative === 'string' ? data.narrative : data.answer));
    const source = ANSWER_SOURCES[data.answer_source] ? data.answer_source : 'model';
    $('answer-source').textContent = ANSWER_SOURCES[source]
      + (data.round_limit_reached ? ' The tool-round limit was reached.' : '');
    $('answer-notice').hidden = source !== 'evidence_summary';
    $('answer-notice-text').textContent = typeof data.answer_notice === 'string'
      ? data.answer_notice : 'No model-generated conclusion is shown.';
    renderAnalysis(data.analysis);
    $('result-id').textContent = data.id;
    renderEvidence(data.evidence);
    renderTimeline(data.tool_trace);
    $('timeline-panel').open = false;
    showOnly(ui.result);
    announce(`Investigation ${limited ? 'incomplete' : 'completed'} with `
      + `${data.evidence.length} evidence records.`);
  }

  /* ---------- Submit ---------- */

  async function readJson(response) {
    try { return await response.json(); } catch { return null; }
  }

  async function submit(event) {
    event.preventDefault();
    if (state.controller) return;
    const question = ui.question.value.trim();
    if (!question) {
      setFieldError('Enter a question to investigate.');
      ui.question.focus();
      return;
    }
    if (question.length > state.config.max_question_length) {
      setFieldError(`Questions are limited to ${state.config.max_question_length} characters.`);
      ui.question.focus();
      return;
    }
    setFieldError('');

    state.controller = new AbortController();
    let aborted = false;
    setBusy(true);
    showLoading();
    try {
      const response = await fetch(API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ question, max_rounds: Number(ui.rounds.value) }),
        signal: state.controller.signal,
      });
      const body = await readJson(response);
      if (response.ok && validResult(body)) renderResult(body);
      else if (response.ok) showError({
        title: 'Unexpected response',
        message: 'The server returned a result in an unexpected format. Check the server logs.',
      });
      else showError(errorFromResponse(response.status, body));
    } catch (err) {
      if (err && err.name === 'AbortError') {
        aborted = true;
        showOnly(ui.notice);
        announce('Stopped waiting for the investigation.');
      } else {
        showError({
          title: 'Cannot reach the OpsPilot server',
          message: 'The request did not complete. Check that the server is running and try again.',
        });
        loadConfig();
      }
    } finally {
      stopTimer();
      state.controller = null;
      setBusy(false);
    }
    if (!aborted) {
      const target = $('results-title');
      target.scrollIntoView({ block: 'start',
        behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      target.focus({ preventScroll: true });
    }
  }

  /* ---------- Wiring ---------- */

  function showView() {
    const view = location.hash === '#how-it-works' ? 'how-it-works' : 'workspace';
    $('view-workspace').hidden = view !== 'workspace';
    $('view-how-it-works').hidden = view !== 'how-it-works';
    document.querySelectorAll('.nav a[data-view]').forEach((link) => {
      if (link.dataset.view === view) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
  }

  ui.form.addEventListener('submit', submit);
  ui.stop.addEventListener('click', () => state.controller && state.controller.abort());
  ui.question.addEventListener('input', () => { updateCounter(); setFieldError(''); });
  ui.question.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) ui.form.requestSubmit();
  });
  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      ui.question.value = chip.dataset.prompt;
      updateCounter();
      setFieldError('');
      ui.question.focus();
    });
  });
  $('toggle-steps').addEventListener('click', () => {
    const steps = document.querySelectorAll('#timeline details');
    const open = Array.from(steps).some((step) => !step.open);
    steps.forEach((step) => { step.open = open; });
    $('timeline-panel').open = true;
    $('toggle-steps').textContent = open ? 'Collapse all steps' : 'Expand all steps';
  });
  $('new-btn').addEventListener('click', () => {
    ui.question.value = '';
    updateCounter();
    showOnly(ui.empty);
    ui.question.focus();
    window.scrollTo({ top: 0 });
  });
  window.addEventListener('hashchange', showView);

  showView();
  applyConfig(FALLBACK_CONFIG, false);
  setRuntime('Checking...', false);
  loadConfig();
})();
