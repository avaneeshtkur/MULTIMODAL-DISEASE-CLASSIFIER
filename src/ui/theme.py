"""
theme.py — Gradio theme + CSS: light clinical surface, deep navy and teal accents.
"""
import gradio as gr

NAVY = "#0e2a47"
TEAL = "#0d9488"


def build_theme():
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        # System fonts only — no Google Fonts request, so the UI works fully offline.
        font=["Inter", "system-ui", "-apple-system", "Segoe UI", "Noto Sans", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    ).set(
        body_background_fill="#f4f7fa",
        block_background_fill="#ffffff",
        block_border_color="#e2e8f0",
        block_shadow="none",
        button_primary_background_fill=NAVY,
        button_primary_background_fill_hover="#16395f",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_dark=NAVY,
        button_primary_background_fill_hover_dark="#16395f",
        button_primary_text_color_dark="#ffffff",
        body_background_fill_dark="#0b1725",
        block_background_fill_dark="#122337",
        block_border_color_dark="#2b4055",
    )


# Theme changes are client-only and preserve all form and screening state.
THEME_JS = """
() => {
  let saved;
  try { saved = localStorage.getItem('mediscan-theme'); } catch (_) {}
  const requested = new URLSearchParams(location.search).get('__theme');
  const dark = (saved || requested || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
  document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
  document.documentElement.classList.toggle('dark', dark);
  document.body.classList.toggle('dark', dark);
}
"""
TOGGLE_THEME_JS = """
() => {
  const dark = !document.documentElement.classList.contains('dark');
  document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
  document.documentElement.classList.toggle('dark', dark);
  document.body.classList.toggle('dark', dark);
  try { localStorage.setItem('mediscan-theme', dark ? 'dark' : 'light'); } catch (_) {}
}
"""

CSS = """
.gradio-container {
  --ms-bg: #f4f7fa; --ms-surface: #ffffff; --ms-ink: #0e2a47;
  --ms-text: #334155; --ms-muted: #64748b; --ms-line: #e2e8f0; --ms-tint: #f0fdfa;
  max-width: 1440px !important; margin: auto; padding: 28px 40px !important;
  color: var(--ms-text); background: var(--ms-bg);
}
.dark .gradio-container {
  --ms-bg: #0b1725; --ms-surface: #122337; --ms-ink: #e2e8f0;
  --ms-text: #cbd5e1; --ms-muted: #94a3b8; --ms-line: #2b4055; --ms-tint: #123934;
  --body-background-fill: var(--ms-bg); --block-background-fill: var(--ms-surface);
  --input-background-fill: var(--ms-surface); --block-border-color: var(--ms-line);
  --body-text-color: var(--ms-text); --block-label-text-color: var(--ms-text);
  --input-text-color: var(--ms-text);
}
.ms-header { display:flex; align-items:center; justify-content:space-between; gap:24px;
  padding: 8px 0 24px; border-bottom: 2px solid #0d9488; }
.ms-header h1 { color:var(--ms-ink); font-size:30px; letter-spacing:-1.2px; line-height:1.2; margin:0; font-weight:700; }
.ms-header p { color:var(--ms-muted); margin:7px 0 0; font-size:13px; }
.ms-wordmark { display:flex; align-items:center; gap:12px; }
.ms-mark { display:grid; place-items:center; width:38px; height:38px; background:#0e2a47; color:white; font-size:26px; font-weight:400; border-radius:6px; }
.ms-header-meta { text-align:right; font-size:12px; color:var(--ms-muted); line-height:1.7; }
.ms-header-meta strong { display:block; color:var(--ms-ink); font-weight:600; }
#ms-toolbar { align-items:center; padding:4px 0 10px; }
#ms-language { border:0; background:transparent; min-width:220px !important; }
#ms-theme { min-width:150px !important; flex-grow:0; }
.gradio-container .tab-container:not(.visually-hidden) { border-bottom:1px solid var(--ms-line) !important; gap:24px; padding:0 !important; }
.gradio-container .tab-container:not(.visually-hidden) button { border:0 !important; border-radius:0 !important; padding:14px 0 !important; background:transparent !important; color:var(--ms-muted) !important; font-weight:500 !important; }
.gradio-container .tab-container:not(.visually-hidden) button.selected { color:var(--ms-ink) !important; box-shadow:inset 0 -2px #0d9488 !important; }
.gradio-container .tabitem { padding:28px 0 !important; border:0 !important; background:transparent !important; animation:ms-enter 180ms ease-out; }
.gradio-container h2 { color:var(--ms-ink); font-size:25px; letter-spacing:-.6px; }
.gradio-container h3, .gradio-container h4 { color:var(--ms-ink); font-weight:600; }
.gradio-container h4 { margin:18px 0 8px; }
.gradio-container button { transition:background-color 140ms, border-color 140ms; box-shadow:none !important; }
.gradio-container button:focus-visible, .gradio-container input:focus-visible { outline:2px solid #0d9488 !important; outline-offset:3px; }
#ms-screening-input { padding-right:30px; border-right:1px solid var(--ms-line); }
#ms-screening-result { padding-left:18px; }
.ms-empty { min-height:180px; display:flex; flex-direction:column; justify-content:center; border-top:2px solid #0d9488; padding:28px; background:var(--ms-surface); }
.ms-empty strong { color:var(--ms-ink); font-size:20px; font-weight:500; }
.ms-empty p { color:var(--ms-muted); font-size:14px; max-width:360px; line-height:1.6; }
.ms-banner { padding:12px 16px; font-size:13px; line-height:1.6; border-left:3px solid #0d9488; color:var(--ms-text); background:var(--ms-tint); }
.ms-banner.warn, .ms-disclaimer { border-left:3px solid #d97706; background:#fffbeb; color:#78350f; }
.ms-banner.sim { background:#fee2e2; color:#991b1b; border-left-color:#991b1b; font-weight:600; }
.ms-disclaimer { padding:14px 16px; font-size:12px; line-height:1.65; margin-top:16px; }
.dark .ms-banner.warn, .dark .ms-disclaimer { background:#332b1d; color:#fde68a; }
.dark .ms-banner.sim { background:#3d2028; color:#fecaca; }
.ms-cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border-top:1px solid var(--ms-line); border-bottom:1px solid var(--ms-line); margin:16px 0 24px; }
.ms-card { padding:20px 20px 20px 0; }
.ms-card .k { color:var(--ms-muted); font-size:12px; }
.ms-card .v { color:var(--ms-ink); font-size:32px; font-weight:500; letter-spacing:-1px; font-variant-numeric:tabular-nums; margin-top:6px; }
.ms-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:24px 32px; padding:12px 0 24px; }
.ms-step { color:var(--ms-muted); font-size:13px; line-height:1.6; }
.ms-step b { color:#0d9488; display:block; margin-bottom:6px; font-weight:600; }
.ms-table-wrap { overflow-x:auto; }
.ms-models { width:100%; border-collapse:collapse; color:var(--ms-text); }
.ms-models td, .ms-models th { padding:14px 12px; border-bottom:1px solid var(--ms-line); text-align:left; font-size:12px; }
.ms-models th { font-weight:500; color:var(--ms-muted); background:var(--ms-surface); white-space:nowrap; }
.ms-models .ok { color:#0d9488; font-weight:600; }
.ms-models .bad { color:#b91c1c; font-weight:600; }
.dark .ms-models .bad { color:#fca5a5; }
.ms-models .miss, .ms-small { color:var(--ms-muted); font-size:12px; line-height:1.6; }
.ms-result { background:var(--ms-surface); color:var(--ms-text); }
.ms-result .head { padding:16px 20px; color:white; font-size:17px; font-weight:600; }
.ms-result .body { padding:20px; }
.ms-result table { width:100%; border-collapse:collapse; font-size:13px; }
.ms-result td { padding:10px 0; border-bottom:1px solid var(--ms-line); vertical-align:top; overflow-wrap:anywhere; }
.ms-result td.k { color:var(--ms-muted); width:42%; padding-right:12px; }
.ms-scorebar { position:relative; height:8px; background:var(--ms-line); margin:12px 0; }
.ms-scorebar .band { position:absolute; top:0; bottom:0; background:#94a3b8; }
.ms-scorebar .thr { position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--ms-ink); }
.ms-scorebar .dot { position:absolute; top:-4px; width:16px; height:16px; border-radius:50%; border:2px solid white; transform:translateX(-50%); }
.ms-section-title { color:var(--ms-ink); font-weight:600; margin:24px 0 12px; }
.ms-two { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:24px; }
.ms-box { border-top:1px solid var(--ms-line); padding:16px 0; color:var(--ms-text); }
.ms-box ul { padding-left:18px; font-size:13px; line-height:1.7; }
.ms-pill { display:inline-block; font-size:12px; padding:3px 8px; background:var(--ms-line); color:var(--ms-ink); }
.ms-pill.PRIORITY { background:#ffedd5; color:#9a3412; }
.ms-banner code, .ms-result code, .ms-box code, .ms-models code { color:var(--ms-ink); background:var(--ms-bg); overflow-wrap:anywhere; }
@keyframes ms-enter { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
@media (prefers-reduced-motion:reduce) { .gradio-container *, .gradio-container *::before { animation:none !important; transition:none !important; } }
@media (max-width:760px) {
  .gradio-container { padding:16px !important; }
  .ms-header { align-items:flex-start; } .ms-header-meta { display:none; }
  .ms-header h1 { font-size:26px; } .ms-header p { font-size:12px; }
  .gradio-container .tab-container:not(.visually-hidden) { gap:18px; flex-wrap:wrap; }
  .ms-cards { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .ms-steps { grid-template-columns:1fr 1fr; gap:20px; }
  #ms-screening-input { padding-right:0; border-right:0; }
  #ms-screening-result { padding-left:0; }
}

/* Workspace composition: persistent rail, focused canvas, imaging reference strip. */
.gradio-container { max-width:none !important; padding:0 40px 40px 268px !important; min-height:100vh; }
#ms-brand-row { height:0; min-height:0; gap:0; }
.ms-rail { position:fixed; inset:0 auto 0 0; width:228px; background:#0e2a47; color:#fff; padding:34px 22px; z-index:20; box-sizing:border-box; }
.ms-wordmark { gap:10px; }
.ms-wordmark h1 { color:#fff; font-size:21px; letter-spacing:-.7px; margin:0; line-height:1.3; white-space:nowrap; }
.ms-mark { width:32px; height:36px; flex-shrink:0; background:#0d9488; border-radius:9px; }
.ms-rail-caption { font-size:8px; letter-spacing:1.8px; color:#94b5c9; }
.ms-rail-label { margin-top:54px; font-size:10px; letter-spacing:1.7px; color:#8da7bf; padding-left:12px; }
.ms-rail-footer { position:absolute; bottom:32px; left:30px; font-size:12px; color:#e2e8f0; line-height:1.8; }
.ms-rail-footer small { display:block; color:#93adc2; font-size:11px; margin-top:10px; line-height:1.7; }
.ms-online { display:inline-block; width:6px; height:6px; border-radius:100%; background:#5eead4; margin-right:7px; }
.ms-topline { height:65px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--ms-line); color:var(--ms-muted); font-size:12px; }
.ms-slash { padding:0 12px; color:#94a3b8; }
.ms-local { font-size:9px; letter-spacing:1.5px; color:#0d9488; }
#ms-toolbar { padding:12px 0; gap:18px; border-bottom:1px solid var(--ms-line); }
#ms-toolbar .ms-banner { padding:7px 12px; font-size:11px; border-radius:5px; }
#ms-language { min-width:200px !important; padding:0; }
#ms-language > .wrap { gap:5px; }
#ms-theme { font-size:11px; min-width:132px !important; border-radius:6px; }
#ms-navigation .tab-container:not(.visually-hidden) { position:fixed; top:175px; left:16px; width:196px; display:flex; flex-direction:column; align-items:stretch; gap:6px; border:0 !important; z-index:21; }
#ms-navigation .tab-container:not(.visually-hidden) button { text-align:left; width:100%; padding:12px 16px !important; border-radius:6px !important; color:#b6c9da !important; font-size:13px; transition:background 160ms,color 160ms; }
#ms-navigation .tab-container:not(.visually-hidden) button:hover { background:#16395f !important; color:#fff !important; }
#ms-navigation .tab-container:not(.visually-hidden) button.selected { background:#1b405b !important; color:#fff !important; box-shadow:inset 3px 0 #2dd4bf !important; }
#ms-navigation > .tabitem { padding-top:28px !important; }
#ms-page-heading { align-items:center; }
.ms-eyebrow { color:#0d9488; font-size:10px; font-weight:650; letter-spacing:1.6px; margin-bottom:10px; }
.ms-page-heading h2 { font-size:32px; font-weight:600; letter-spacing:-1.1px; margin:0; line-height:1.2; color:var(--ms-ink); }
.ms-page-heading p { color:var(--ms-muted); font-size:13px; margin:10px 0 4px; }
.ms-cards { grid-template-columns:repeat(8,minmax(0,1fr)); background:var(--ms-surface); border:1px solid var(--ms-line); border-radius:8px; padding:18px 0; margin:14px 0 10px; }
.ms-card { padding:0 16px; border-right:1px solid var(--ms-line); }
.ms-card:last-child { border:0; }
.ms-card .k { font-size:10px; min-height:28px; line-height:1.35; }
.ms-card .v { font-size:28px; font-weight:600; margin:0; }
.ms-card:first-child .v { color:#0d9488; }
.ms-section-heading { display:flex; justify-content:space-between; align-items:center; margin:14px 0 0; }
.ms-section-heading h3 { font-size:16px; margin:0; }
.ms-section-heading > span { color:var(--ms-muted); font-size:11px; }
.ms-modalities { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:22px; margin:8px 0 12px; }
.ms-modality { min-width:0; }
.ms-modality-image { position:relative; height:176px; overflow:hidden; background:#091827; border-radius:8px; }
.ms-modality-image img { width:100%; height:100%; object-fit:cover; transition:transform 300ms ease; }
.ms-modality:first-child img { object-fit:contain; }
.ms-modality:hover img { transform:scale(1.045); }
.ms-image-label { position:absolute; bottom:10px; left:12px; color:#fff; background:#0e2a47; padding:4px 6px; font-size:8px; letter-spacing:1px; border-radius:3px; }
.ms-modality-no { position:absolute; top:10px; right:12px; font-size:11px; color:white; background:#0e2a47; padding:3px 6px; border-radius:3px; }
.ms-modality-info { padding:14px 0 0; }
.ms-modality-info > span { font-size:10px; text-transform:uppercase; letter-spacing:1.1px; color:var(--ms-muted); }
.ms-modality-info h3 { font-size:16px; margin:5px 0 7px; letter-spacing:-.3px; }
.ms-model-state { font-size:11px; color:#0d9488; margin:0; }
.ms-models th { font-size:10px; letter-spacing:.6px; text-transform:uppercase; padding:10px 12px; }
.ms-models td { font-size:11px; padding:12px; }
.ms-table-wrap { background:var(--ms-surface); border:1px solid var(--ms-line); border-radius:8px; }
#ms-screening-input { background:var(--ms-surface); padding:20px 24px; border:1px solid var(--ms-line); border-radius:10px; min-width:310px !important; }
#ms-screening-input h4 { font-size:14px; border-bottom:1px solid var(--ms-line); padding-bottom:12px; margin:12px 0 4px; }
#ms-screening-result { padding:0 0 0 16px; min-width:310px !important; }
#ms-screening-result h4 { margin-top:0; font-size:14px; }
#ms-module { border:0; padding:4px 0; }
#ms-module label { font-size:12px; }
#ms-upload { border:1px dashed #80b7b4; background:var(--ms-tint); border-radius:8px; }
.ms-empty { min-height:285px; align-items:center; text-align:center; border:1px solid var(--ms-line); border-radius:10px; padding:28px 20px 20px; }
.ms-empty strong { font-size:22px; letter-spacing:-.5px; font-weight:600; }
.ms-empty p { font-size:12px; max-width:310px; }
.ms-empty-foot { border-top:1px solid var(--ms-line); padding-top:18px; margin-top:16px; width:100%; font-size:10px; color:var(--ms-muted); }
.ms-empty-foot span { padding:0 8px; color:#0d9488; }
.ms-scan-frame { width:54px; height:54px; position:relative; border:1px solid #0d9488; border-radius:10px; margin:0 0 22px; display:grid; place-items:center; background:var(--ms-tint); }
.ms-scan-frame::before, .ms-scan-frame::after { content:""; position:absolute; background:#0d9488; border-radius:2px; }
.ms-scan-frame::before { width:24px; height:1px; }
.ms-scan-frame::after { height:24px; width:1px; }
.ms-disclaimer { border-radius:6px; }
@media(min-width:1500px) { .gradio-container { padding-right:70px !important; padding-left:298px !important; } .ms-modality-image { height:210px; } }
@media(max-width:1150px) { .ms-cards { grid-template-columns:repeat(4,minmax(0,1fr)); gap:18px 0; } .ms-card:nth-child(4n) { border:0; } .ms-modality-image { height:145px; } }
@media(max-width:800px) {
 .gradio-container { padding:0 18px 24px !important; }
 #ms-brand-row { height:auto; }
 .ms-rail { position:relative; width:auto; padding:18px; margin:0 -18px; }
 .ms-rail-label,.ms-rail-footer { display:none; }
 .ms-wordmark h1 { font-size:22px; }
 .ms-topline { height:42px; } .ms-local { display:none; }
 #ms-navigation .tab-container:not(.visually-hidden) { position:relative; inset:auto; width:auto; flex-direction:row; flex-wrap:wrap; gap:4px; background:#0e2a47; padding:6px !important; border-radius:7px; }
 #ms-navigation .tab-container:not(.visually-hidden) button { width:auto; padding:10px !important; font-size:11px; }
 #ms-navigation .tab-container:not(.visually-hidden) button.selected { box-shadow:inset 0 -2px #2dd4bf !important; }
 .ms-modalities { gap:12px; } .ms-modality-image { height:130px; }
 .ms-modality-info h3 { font-size:13px; } .ms-modality-info > span { font-size:8px; }
 .ms-section-heading > span { display:none; } .ms-page-heading h2 { font-size:27px; }
 #ms-screening-result { padding:0; }
}
@media(max-width:480px) { .ms-modalities { grid-template-columns:1fr; } .ms-modality { display:grid; grid-template-columns:110px 1fr; gap:16px; } .ms-modality-image { height:110px; } .ms-card { padding:0 10px; } .ms-card .v { font-size:23px; } }

.gradio-container .html-container { padding:0 !important; }
#ms-brand-row { display:contents; }
#ms-navigation > .tab-wrapper { height:0; margin:0; padding:0; }
#ms-navigation .tab-container:not(.visually-hidden) { height:auto; overflow:visible; background:none; }
#ms-navigation .tab-container::after, #ms-navigation .tab-container button::after { display:none; }
#ms-navigation .tab-container button { height:auto; flex-shrink:0; }
#ms-language { background:transparent !important; border:0 !important; }
#ms-language label { padding:6px 10px; font-size:11px; border-radius:5px; }
#ms-toolbar { min-height:50px; }
.gradio-container .main { padding-top:0 !important; }
@media(max-width:800px) { #ms-navigation > .tab-wrapper { height:auto; margin-bottom:8px; } }
"""
