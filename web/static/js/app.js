const $ = id => document.getElementById(id);
let demoCases = [];
let selectedFile = null, previewUrl = null, result = null, busy = false, imageRevision = 0;
const message = (text, state = '') => { $('status-message').textContent = text; $('status-message').className = state; };
const showImage = (id, src) => { $(id).hidden = !src; if (src) $(id).src = src; else $(id).removeAttribute('src'); };
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) { const body = await response.json().catch(() => ({})); const error = new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}). Please try again.`); error.status = response.status; throw error; }
  return response.json();
}
$('theme-toggle').onclick = () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('mediscan-theme', theme); } catch {}
};
$('menu-toggle').onclick = () => {
  const open = $('navigation').classList.toggle('open'); $('menu-toggle').setAttribute('aria-expanded', String(open));
};
$('navigation').onclick = event => { if (event.target.closest('a')) { $('navigation').classList.remove('open'); $('menu-toggle').setAttribute('aria-expanded', 'false'); } };
function clearResult() {
  $('download-report').hidden = true; $('download-report').removeAttribute('href'); $('report-status').textContent = ''; $('image-quality').textContent = '';
  result = null; $('screening-result').replaceChildren(); $('result-metrics').innerHTML = '<p class="small">Screen an image to see measured results.</p>';
  showImage('edge-preview', null); $('edge-empty').hidden = false; $('download-result').hidden = true;
  $('comparison-results').hidden = true; $('comparison-empty').hidden = false; $('comparison-images').replaceChildren();
  $('compare-button').disabled = true; $('export-csv').disabled = true; $('viewer-status').textContent = selectedFile ? 'Ready to screen' : 'Awaiting input';
  $('result-settings').textContent = 'Model attention is not a lesion boundary or a diagnosis.';
  $('result-algorithm').textContent = 'OUTPUT'; $('export-help').textContent = 'Run a screening to enable CSV export.'; message('');
}
function selectFile(file) {
  if (busy) return;
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) { message('Select an image smaller than 10 MB.', 'error'); return; }
  if (!/\.(jpe?g|png|bmp)$/i.test(file.name)) { message('Please select a JPEG, PNG, or BMP image.', 'error'); return; }
  imageRevision++; selectedFile = file; if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = URL.createObjectURL(file);
  clearResult(); showImage('original-preview', previewUrl); $('original-empty').hidden = true; $('remove').hidden = false;
  $('file-info').textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`; $('input-size').textContent = 'INPUT';
}
$('original-preview').onload = () => { $('input-size').textContent = `${$('original-preview').naturalWidth} × ${$('original-preview').naturalHeight}`; };
$('original-preview').onerror = () => { message('The image preview could not be decoded. Choose another file.', 'error'); };
$('image-file').onchange = event => selectFile(event.target.files[0]);
for (const type of ['dragenter', 'dragover']) $('dropzone').addEventListener(type, event => { event.preventDefault(); if (!busy) $('dropzone').classList.add('dragover'); });
for (const type of ['dragleave', 'drop']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));
function removeImage() {
  if (busy) return;
  imageRevision++; selectedFile = null; if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null;
  $('image-file').value = ''; showImage('original-preview', null); $('original-empty').hidden = false; $('remove').hidden = true;
  $('file-info').textContent = 'No image selected. Try an example to begin.'; $('input-size').textContent = 'INPUT'; clearResult();
}
$('remove').onclick = removeImage;
$('reset-button').onclick = () => { if (busy) return; $('studio-form').reset(); $('report-name').readOnly = false; $('demo-details').textContent = ''; removeImage(); updateModule(); updateSampleLabel(); };
function updateModule() {
  $('algorithm-help').textContent = 'Choose the disease specialist. Technical quality and basic image-type suitability are checked first; accepted images go directly to that specialist, which produces the disease score. The legacy modality classifier does not affect scoring.';
}
$('algorithm').onchange = () => { clearResult(); updateModule(); };
$('patient').onchange = async () => { ++imageRevision; clearResult(); const d = demoCases.find(d => String(d.patient_id) === $('patient').value); if (d) { try { await loadDemo(d); } catch (error) { message(error.message, 'error'); } } else { $('demo-details').textContent = ''; $('report-name').value = ''; $('report-name').readOnly = false; } };
$('explain').onchange = clearResult;
$('report-name').onchange = clearResult;
function updateSampleLabel() { $('sample-button').textContent = `Use ${$('sample-select').selectedOptions[0].textContent.toLowerCase()} sample ↗`; }
$('sample-select').onchange = updateSampleLabel;
$('sample-button').onclick = async () => {
  if (busy) return;
  const chosen = demoCases.find(d => `demo-${d.patient_id}` === $('sample-select').value);
  if (chosen) { try { $('patient').value = chosen.patient_id; await loadDemo(chosen); } catch (error) { message(error.message, 'error'); } return; }
  const revision = ++imageRevision, module = $('sample-select').value;
  try {
    const response = await fetch(`/api/samples/${module}`); if (!response.ok) throw new Error('Sample unavailable. Generate demo assets or upload an image.');
    const blob = await response.blob(); if (revision !== imageRevision || busy) return;
    selectFile(new File([blob], `sample_${module}.${module === 'oral' ? 'jpg' : 'png'}`, {type: blob.type}));
  } catch (error) { message(error.message, 'error'); }
};
async function loadDemo(d) {
  const revision = ++imageRevision;
  const response = await fetch(`/api/demos/${d.patient_id}/scan`);
  if (!response.ok) throw new Error('Saved scan unavailable.');
  const blob = await response.blob();
  if (revision !== imageRevision || busy) return;
  selectFile(new File([blob], d.filename));
  $('report-name').value = d.name; $('report-name').readOnly = true;
  $('algorithm').value = d.module; updateModule();
  $('demo-details').textContent = `${d.name} · reference: ${d.expected_label}. ${d.details} Source: ${d.source}. Reference label is not a model result.`;
}
async function loadPatients(selected) {
  const [patients, demos] = await Promise.all([api('/api/patients'), api('/api/demos')]); demoCases = demos;
  $('patient').replaceChildren(new Option('Select a patient', ''));
  for (const p of patients) { const d = demos.find(d => d.patient_id === p.id); $('patient').add(new Option(d ? d.name : `#${p.id} · ${p.reference || 'No reference'}${p.is_demo ? ' · DEMO' : ''}`, p.id)); }
  const previousSample = $('sample-select').value;
  $('sample-select').replaceChildren();
  for (const d of demos) $('sample-select').add(new Option(d.name, `demo-${d.patient_id}`));
  for (const [key, label] of [['tb','Chest X-ray'], ['dr','Retinal fundus'], ['oral','Histopathology']]) $('sample-select').add(new Option(label, key));
  if ([...$('sample-select').options].some(o => o.value === previousSample)) $('sample-select').value = previousSample;
  updateSampleLabel();
  if (selected) $('patient').value = selected;
}
$('new-patient').onclick = () => $('demo-dialog').showModal();
$('demo-cancel').onclick = () => $('demo-dialog').close();
$('demo-form').onsubmit = async event => {
  event.preventDefault(); const button = event.submitter; button.disabled = true; $('demo-error').textContent = '';
  try {
    const patient = await api('/api/patients/demo', {method:'POST', body:new FormData($('demo-form'))});
    await loadPatients(patient.id); await loadDemo(demoCases.find(d => d.patient_id === patient.id));
    $('demo-dialog').close(); $('demo-form').reset(); message('Demo patient and scan saved. Select this record any time to reuse the scan.');
  } catch (error) { $('demo-error').textContent = error.message; }
  finally { button.disabled = false; }
};
function setBusy(value) {
  busy = value; $('studio-form').setAttribute('aria-busy', String(value));
  $('studio-form').querySelectorAll('input,select,button').forEach(control => control.disabled = value);
  $('compare-button').disabled = value || !result?.images?.overlay;
  $('run-button').textContent = value ? 'Screening…' : 'Run screening →';
}
function renderMetrics(metrics) {
  const entries = [['Processing time', metrics.processing_time_ms == null ? '—' : `${Math.round(metrics.processing_time_ms)} ms`], ['Disease model score', metrics.model_score == null ? '—' : `${(metrics.model_score * 100).toFixed(2)}%`], ['Threshold', metrics.threshold == null ? '—' : `${(metrics.threshold * 100).toFixed(2)}%`], ['Module', (metrics.disease || '—').toUpperCase()], ['Provider', metrics.provider || '—'], ['Disease target', metrics.target_definition || '—']];
  $('result-metrics').replaceChildren();
  for (const [label,value] of entries) { const div = document.createElement('div'); div.className = 'metric'; const strong = document.createElement('strong'); strong.textContent = value; const span = document.createElement('span'); span.textContent = label; div.append(strong,span); $('result-metrics').append(div); }
}
function renderComparison(images) {
  $('comparison-images').replaceChildren();
  for (const [label,src] of [['Original',previewUrl],['Model input',images.base],['Attention heatmap',images.heatmap],['Overlay',images.overlay]]) {
    if (!src) continue;
    const figure = document.createElement('figure'), caption = document.createElement('figcaption'), img = document.createElement('img'); caption.textContent = label.toUpperCase(); img.src = src; img.alt = label; figure.append(caption,img); $('comparison-images').append(figure);
  }
  $('comparison-results').hidden = false; $('comparison-empty').hidden = true;
}
$('studio-form').onsubmit = async event => {
  event.preventDefault(); if (busy) return;
  if (!selectedFile) return message('Upload a medical image or choose a sample first.', 'error');
  if (!$('patient').value) return message('Select a patient or create a demo patient first.', 'error');
  clearResult(); const data = new FormData(); data.append('patient_name', $('report-name').value); data.append('image',selectedFile); data.append('patient_id',$('patient').value); data.append('module',$('algorithm').value); data.append('explain',$('explain').checked); data.append('allow_rescreen',$('rescreen').checked);
  setBusy(true); message('Validating image, running the specialist, and generating the explanation and PDF report…', 'loading'); $('viewer-status').textContent = 'Processing';
  try {
    const job = await api('/api/jobs', {method:'POST',body:data,headers:{'Idempotency-Key':crypto.randomUUID()}});
    try { sessionStorage.setItem('mediscan-job', job.id); } catch {}
    await followJob(job.id);
  } catch (error) { message(error.message, 'error'); $('viewer-status').textContent = 'Unable to complete'; }
  finally { setBusy(false); }
};
async function followJob(id) {
  async function readJob() {
    for (let attempt = 0; ; attempt++) {
      try { return await api(`/api/jobs/${id}`); }
      catch (error) {
        if (attempt >= 2 || (error.status && error.status < 500 && error.status !== 429)) throw error;
        message('Reconnecting to the saved screening job…', 'loading');
        await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
      }
    }
  }
  let job = await readJob();
  while (['queued', 'running'].includes(job.status)) {
    message(job.status === 'queued' ? 'Screening queued…' : 'Specialist analysis in progress…', 'loading');
    $('viewer-status').textContent = job.status;
    await new Promise(resolve => setTimeout(resolve, 1000));
    job = await readJob();
  }
  try { sessionStorage.removeItem('mediscan-job'); } catch {}
  if (!job.result) throw new Error(job.error || `Screening ${job.status}. No disease decision was produced.`);
  result = job.result;
    // These fragments come from the existing server renderer, which escapes dynamic content.
    $('screening-result').innerHTML = result.html + result.clinical;
    $('viewer-status').textContent = result.status.replaceAll('_',' '); renderMetrics(result.metrics);
    $('download-report').href = result.report_url; $('download-report').hidden = !result.report_url;
    $('report-status').textContent = result.report_error || 'PDF generated automatically. Automated screening findings require clinical review; this is not a confirmed diagnosis.';
    const q = result.quality, qm = q?.metrics || {};
    $('image-quality').textContent = q ? `Image quality: ${q.status.replaceAll('_', ' ')} · Sharpness: ${qm.blur_laplacian_var ?? 'not measured'} · Exposure p1/p99: ${qm.p1 ?? '—'} / ${qm.p99 ?? '—'} (0–255). ${q.note} ${(q.warnings || []).join(' ')}` : '';
    const overlay = result.images.overlay;
    showImage('edge-preview',overlay); $('edge-empty').hidden = !!overlay;
    message(job.status === 'failed' ? 'Analysis unavailable. No disease decision was produced.' : result.stage === 'rejected' ? 'Image not assessed. Review the reason and next step below.' : overlay ? 'Screening complete. Review the result and explanation below.' : `Screening complete. ${result.explanation_message}`, ['failed','ungradable','unsupported'].includes(job.status) ? 'error' : '');
    if (overlay) { $('download-result').href = overlay; $('download-result').download = 'mediscan-gradcam.png'; $('download-result').hidden = false; renderComparison(result.images); }
    $('result-algorithm').textContent = overlay ? 'GRAD-CAM' : 'UNAVAILABLE';
    $('result-settings').textContent = result.metrics.model_version ? `Model ${result.metrics.model_version}${result.metrics.simulated ? ' · SIMULATED' : ''} · Screening aid only` : 'No specialist result';
    $('export-csv').disabled = result.stage !== 'screened'; $('export-help').textContent = result.stage === 'screened' ? 'Exports the current result and model provenance. Use the report link for the patient PDF.' : 'Rejected uploads have no screening measurements.';
}
$('compare-button').onclick = () => $('comparison').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
$('export-csv').onclick = () => {
  if (!result || result.stage !== 'screened') return;
  const values = {result_id:result.result_id,status:result.status,...result.metrics};
  const quote = value => `"${String(value ?? '').replaceAll('"','""')}"`;
  const blob = new Blob([Object.keys(values).map(quote).join(',')+'\n'+Object.values(values).map(quote).join(',')+'\n'],{type:'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = 'mediscan-measurements.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
};
api('/api/status').then(data => {
  const ready = data.models.filter(m => ['available','configured'].includes(m.status)).length;
  $('backend-status').textContent = data.simulation ? 'SIMULATION MODE' : `${ready}/${data.models.length} models available or configured · ${data.processing}`;
  $('research-notice').hidden = !data.research_evaluation;
  if (data.processing === 'remote') $('algorithm-help').textContent += ' Configured remote specialists receive scan pixels; patient names remain local.';
  if (!data.legacy_workspace) document.querySelectorAll('a[href^="/workspace"]').forEach(a => a.hidden = true);
}).catch(() => { $('backend-status').textContent = 'Engine unavailable'; });
loadPatients().catch(error => message(`Unable to load patients. ${error.message}`, 'error'));
try {
  const pending = sessionStorage.getItem('mediscan-job');
  if (pending && /^[a-f0-9]{32}$/.test(pending)) {
    setBusy(true);
    followJob(pending).catch(error => message(error.message, 'error')).finally(() => setBusy(false));
  }
} catch {}
