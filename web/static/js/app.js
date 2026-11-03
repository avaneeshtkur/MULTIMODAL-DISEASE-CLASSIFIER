const $ = id => document.getElementById(id);
let selectedFile = null, previewUrl = null, result = null, busy = false, imageRevision = 0;
const message = (text, state = '') => { $('status-message').textContent = text; $('status-message').className = state; };
const showImage = (id, src) => { $(id).hidden = !src; if (src) $(id).src = src; else $(id).removeAttribute('src'); };
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}). Please try again.`); }
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
$('reset-button').onclick = () => { if (busy) return; $('studio-form').reset(); removeImage(); updateModule(); updateSampleLabel(); };
function updateModule() {
  $('algorithm-help').textContent = $('algorithm').value === 'auto' ? 'Automatic routing uses the trained modality gatekeeper. Manual selection never skips image validation.' : 'Manual routing selected. Image quality and modality checks still apply.';
}
$('algorithm').onchange = () => { clearResult(); updateModule(); };
$('patient').onchange = clearResult;
$('explain').onchange = clearResult;
function updateSampleLabel() { $('sample-button').textContent = `Use ${$('sample-select').selectedOptions[0].textContent.toLowerCase()} sample ↗`; }
$('sample-select').onchange = updateSampleLabel;
$('sample-button').onclick = async () => {
  if (busy) return;
  const revision = ++imageRevision, module = $('sample-select').value;
  try {
    const response = await fetch(`/api/samples/${module}`); if (!response.ok) throw new Error('Sample unavailable. Generate demo assets or upload an image.');
    const blob = await response.blob(); if (revision !== imageRevision || busy) return;
    selectFile(new File([blob], `sample_${module}.${module === 'oral' ? 'jpg' : 'png'}`, {type: blob.type}));
  } catch (error) { message(error.message, 'error'); }
};
async function loadPatients(selected) {
  const patients = await api('/api/patients'); $('patient').replaceChildren(new Option('Select a patient', ''));
  for (const p of patients) $('patient').add(new Option(`#${p.id} · ${p.reference || 'No reference'}${p.is_demo ? ' · DEMO' : ''}`, p.id));
  if (selected) $('patient').value = selected;
}
$('new-patient').onclick = async () => {
  $('new-patient').disabled = true;
  try { const patient = await api('/api/patients/demo', {method: 'POST'}); await loadPatients(patient.id); clearResult(); message(`Demo patient #${patient.id} created. Screenings will be saved as demo records.`); }
  catch (error) { message(error.message, 'error'); }
  finally { $('new-patient').disabled = false; }
};
function setBusy(value) {
  busy = value; $('studio-form').setAttribute('aria-busy', String(value));
  $('studio-form').querySelectorAll('input,select,button').forEach(control => control.disabled = value);
  $('compare-button').disabled = value || !result?.images?.overlay;
  $('run-button').textContent = value ? 'Screening…' : 'Run screening →';
}
function renderMetrics(metrics) {
  const entries = [['Processing time', metrics.processing_time_ms == null ? '—' : `${Math.round(metrics.processing_time_ms)} ms`], ['Model score', metrics.model_score?.toFixed(3) ?? '—'], ['Threshold', metrics.threshold?.toFixed(3) ?? '—'], ['Module', (metrics.disease || '—').toUpperCase()]];
  $('result-metrics').replaceChildren();
  for (const [label,value] of entries) { const div = document.createElement('div'); div.className = 'metric'; const strong = document.createElement('strong'); strong.textContent = value; const span = document.createElement('span'); span.textContent = label; div.append(strong,span); $('result-metrics').append(div); }
}
function renderComparison(images) {
  $('comparison-images').replaceChildren();
  for (const [label,src] of [['Original',previewUrl],['Model input',images.base],['Attention heatmap',images.heatmap],['Overlay',images.overlay]]) {
    const figure = document.createElement('figure'), caption = document.createElement('figcaption'), img = document.createElement('img'); caption.textContent = label.toUpperCase(); img.src = src; img.alt = label; figure.append(caption,img); $('comparison-images').append(figure);
  }
  $('comparison-results').hidden = false; $('comparison-empty').hidden = true;
}
$('studio-form').onsubmit = async event => {
  event.preventDefault(); if (busy) return;
  if (!selectedFile) return message('Upload a medical image or choose a sample first.', 'error');
  if (!$('patient').value) return message('Select a patient or create a demo patient first.', 'error');
  clearResult(); const data = new FormData(); data.append('image',selectedFile); data.append('patient_id',$('patient').value); data.append('module',$('algorithm').value); data.append('explain',$('explain').checked); data.append('allow_rescreen',$('rescreen').checked);
  setBusy(true); message('Validating image, running the specialist, and preparing the explanation…', 'loading'); $('viewer-status').textContent = 'Processing';
  try {
    result = await api('/api/screen', {method:'POST',body:data});
    // These fragments come from the existing server renderer, which escapes dynamic content.
    $('screening-result').innerHTML = result.html + result.clinical;
    $('viewer-status').textContent = result.status.replaceAll('_',' '); renderMetrics(result.metrics);
    const overlay = result.images.overlay;
    showImage('edge-preview',overlay); $('edge-empty').hidden = !!overlay;
    message(result.stage === 'rejected' ? 'Image rejected. Review the reason and next step below.' : overlay ? 'Screening complete. Review the result and explanation below.' : `Screening complete. ${result.explanation_message}`, result.stage === 'rejected' ? 'error' : '');
    if (overlay) { $('download-result').href = overlay; $('download-result').download = 'mediscan-gradcam.png'; $('download-result').hidden = false; renderComparison(result.images); }
    $('result-algorithm').textContent = overlay ? 'GRAD-CAM' : 'UNAVAILABLE';
    $('result-settings').textContent = result.metrics.model_version ? `Model ${result.metrics.model_version}${result.metrics.simulated ? ' · SIMULATED' : ''} · Screening aid only` : 'No specialist result';
    $('export-csv').disabled = result.stage !== 'screened'; $('export-help').textContent = result.stage === 'screened' ? 'Exports the current result. Full history and PDF reports are in the clinical workspace.' : 'Rejected uploads have no screening measurements.';
  } catch (error) { message(error.message, 'error'); $('viewer-status').textContent = 'Unable to complete'; }
  finally { setBusy(false); }
};
$('compare-button').onclick = () => $('comparison').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
$('export-csv').onclick = () => {
  if (!result || result.stage !== 'screened') return;
  const values = {result_id:result.result_id,status:result.status,...result.metrics};
  const quote = value => `"${String(value ?? '').replaceAll('"','""')}"`;
  const blob = new Blob([Object.keys(values).map(quote).join(',')+'\n'+Object.values(values).map(quote).join(',')+'\n'],{type:'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = 'mediscan-measurements.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
};
api('/api/status').then(data => { const ready = data.models.filter(m => m.status === 'available').length; $('backend-status').textContent = data.simulation ? 'SIMULATION MODE' : `${ready}/${data.models.length} models available · CPU`; }).catch(() => { $('backend-status').textContent = 'Engine unavailable'; });
loadPatients().catch(error => message(`Unable to load patients. ${error.message}`, 'error'));
