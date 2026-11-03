"""
translations.py — fixed English/Hindi dictionary.

No machine translation is used at runtime: every user-facing clinical string
is written here so it can be reviewed. Hindi strings are pending review by a
native-speaking clinician (see docs/CLINICAL_LIMITATIONS.md).

t(key, lang, **kw) formats placeholders safely; a missing Hindi key falls
back to English (tests assert none are missing).
"""
from __future__ import annotations

import string

LANGS = {"en": "English", "hi": "हिन्दी"}

EN: dict[str, str] = {
    # ── General ──────────────────────────────────────────────────
    "disclaimer": "This is an experimental AI screening aid, not a medical diagnosis. Results must be reviewed "
                  "by a qualified healthcare professional and confirmed through appropriate clinical testing.",
    "simulated_banner": "SIMULATED OUTPUT — no trained model was run. For interface rehearsal only; "
                        "not a screening result.",
    "score_note": "The model score is the model's raw output, not a calibrated probability of disease.",
    "gradcam_note": "Grad-CAM shows where the model looked. It does not prove the result is correct.",

    # ── Diseases ─────────────────────────────────────────────────
    "disease.tb": "Tuberculosis (chest X-ray)",
    "disease.dr": "Diabetic retinopathy (retinal fundus photograph)",
    "disease.oral": "Oral cancer — OSCC (histopathology)",
    "modality.chest_xray": "Chest X-ray",
    "modality.fundus": "Retinal fundus photograph",
    "modality.histopathology": "Oral histopathology",
    "routing.manual_selection": "selected manually (not verified by a trained classifier)",
    "routing.selected_specialist": "sent directly to the selected disease specialist after suitability checks",
    "routing.trained_gatekeeper": "detected automatically by the trained gatekeeper model",
    "routing.manual_selection+trained_gatekeeper": "selected manually and confirmed by the gatekeeper model",

    # ── Screening statuses ───────────────────────────────────────
    "status.SCREENING_FLAGGED": "Screening flag raised — further evaluation recommended",
    "status.NO_FLAG_DETECTED": "No screening flag detected — this does not rule out disease",
    "status.INCONCLUSIVE": "Inconclusive — no reliable screening result",
    "status.REJECTED": "Image rejected — not analysed",

    # ── Inconclusive / model reasons ─────────────────────────────
    "reason.WITHIN_UNCERTAINTY_BAND": "The model score is too close to the decision threshold to give a reliable result.",
    "reason.OUT_OF_DISTRIBUTION": "The image is outside the type of images the model was trained on.",
    "reason.PREPROCESSING_ERROR": "The image could not be prepared in the way the model requires.",
    "reason.INFERENCE_ERROR": "The model failed while analysing the image.",
    "reason.MODEL_OUTPUT_INVALID": "The model produced an invalid output, so no result is shown.",
    "reason.MODEL_UNAVAILABLE": "The model for this module is not installed or failed validation.",

    # ── Rejections: why / fix ────────────────────────────────────
    "reject.IMAGE_CORRUPTED.why": "The file could not be read as a valid image (it may be damaged or incomplete).",
    "reject.IMAGE_CORRUPTED.fix": "Re-export or re-capture the image and upload it again.",
    "reject.IMAGE_UNSUPPORTED_FORMAT.why": "File format {fmt} is not supported.",
    "reject.IMAGE_UNSUPPORTED_FORMAT.fix": "Upload a PNG or JPEG image (export DICOM studies to PNG first).",
    "reject.IMAGE_TOO_LARGE.why": "The file exceeds the {limit_mb} MB limit or has too many pixels.",
    "reject.IMAGE_TOO_LARGE.fix": "Resize or compress the image and try again.",
    "reject.IMAGE_TOO_SMALL.why": "The image is {width}×{height} px; at least {min_side} px on each side is required.",
    "reject.IMAGE_TOO_SMALL.fix": "Upload the original, full-resolution image.",
    "reject.IMAGE_EXTREME_ASPECT.why": "The image shape is unusual (aspect ratio {ratio}:1, maximum {max_ratio}:1); "
                                       "it may be cropped or a screenshot.",
    "reject.IMAGE_EXTREME_ASPECT.fix": "Upload the complete, uncropped image.",
    "reject.IMAGE_TOO_BLURRY.why": "The image is too blurry for analysis (sharpness {score}, minimum {minimum}).",
    "reject.IMAGE_TOO_BLURRY.fix": "Re-capture the image in focus, keeping the camera or slide steady.",
    "reject.IMAGE_OVEREXPOSED.why": "The image is overexposed (washed out); important detail may be lost.",
    "reject.IMAGE_OVEREXPOSED.fix": "Reduce exposure or brightness and re-capture the image.",
    "reject.IMAGE_UNDEREXPOSED.why": "The image is too dark to contain usable detail.",
    "reject.IMAGE_UNDEREXPOSED.fix": "Increase illumination or exposure and re-capture the image.",
    "reject.DUPLICATE_IMAGE.why": "An identical file was already screened ({where}, {when}). Identical files do not "
                                  "prove that images belong to the same patient.",
    "reject.DUPLICATE_IMAGE.fix": "Check that you selected the right image. To screen it again deliberately, "
                                  "tick 'Allow re-screening of an identical file'.",
    "reject.UNSUPPORTED_MODALITY.why": "This image type is not supported ({label}).",
    "reject.UNSUPPORTED_MODALITY.fix": "Upload a chest X-ray, a retinal fundus photograph or an oral histopathology image.",
    "reject.UNCERTAIN_MODALITY.why": "The image type could not be determined reliably ({reason}).",
    "reject.UNCERTAIN_MODALITY.fix": "Select the screening module manually, or upload a clearer image of a supported type.",
    "reject.MODALITY_MISMATCH.why": "The image does not look like the selected type (expected {expected}, found {observed}).",
    "reject.MODALITY_MISMATCH.fix": "Check the selected module or upload the correct image.",
    "reject.MODEL_UNAVAILABLE.why": "The {module} model is not installed or failed validation.",
    "reject.MODEL_UNAVAILABLE.fix": "Ask the administrator to import the trained model artifacts.",
    "reject.DISEASE_SELECTION_REQUIRED.why": "Choose the disease specialist before screening; automatic disease routing is disabled.",
    "reject.DISEASE_SELECTION_REQUIRED.fix": "Select Tuberculosis, Diabetic retinopathy, or Oral cancer to send the image to that specialist.",

    # ── Referral priority ────────────────────────────────────────
    "priority.ROUTINE": "Routine",
    "priority.PRIORITY": "Priority referral",
    "source.image": "Image model",
    "source.questionnaire": "Questionnaire",

    # ── Recommendations ──────────────────────────────────────────
    "rec.tb.SCREENING_FLAGGED.1": "Refer to a qualified clinician (Medical Officer) for clinical assessment.",
    "rec.tb.SCREENING_FLAGGED.2": "Arrange sputum testing, preferably a WHO-recommended molecular test "
                                  "(NAAT such as CBNAAT/Truenat), as per national TB programme guidelines.",
    "rec.tb.SCREENING_FLAGGED.3": "The chest X-ray should be reviewed by a radiologist or clinician.",
    "rec.tb.NO_FLAG_DETECTED.1": "No TB-related screening flag on this image. This does not rule out tuberculosis.",
    "rec.tb.NO_FLAG_DETECTED.2": "If cough for 2 weeks or more, fever, night sweats or weight loss are present, "
                                 "refer for clinical evaluation and sputum testing regardless of this result.",
    "rec.dr.SCREENING_FLAGGED.1": "Refer to an eye specialist (ophthalmologist) for assessment.",
    "rec.dr.SCREENING_FLAGGED.2": "A dilated retinal examination is recommended.",
    "rec.dr.SCREENING_FLAGGED.3": "Optical coherence tomography (OCT) when clinically indicated, "
                                  "for example if macular oedema is suspected.",
    "rec.dr.NO_FLAG_DETECTED.1": "No diabetic retinopathy screening flag on this image. "
                                 "This does not rule out diabetic retinopathy.",
    "rec.dr.NO_FLAG_DETECTED.2": "Continue regular diabetic eye examinations as advised by the treating clinician.",
    "rec.oral.SCREENING_FLAGGED.1": "A qualified pathologist must review this slide; this flag is not a "
                                    "histopathological diagnosis.",
    "rec.oral.SCREENING_FLAGGED.2": "Refer to a specialist (oral & maxillofacial surgery, ENT or head-and-neck "
                                    "oncology) for oral examination and clinical correlation.",
    "rec.oral.SCREENING_FLAGGED.3": "Further biopsy or tissue sampling only if advised by the specialist.",
    "rec.oral.NO_FLAG_DETECTED.1": "No OSCC screening flag on this image. This does not rule out oral cancer; "
                                   "the slide still requires pathologist review.",
    "rec.oral.NO_FLAG_DETECTED.2": "Mouth ulcers, white or red patches, or lumps lasting more than 2 weeks need "
                                   "clinical examination regardless of this result.",
    "rec.any.INCONCLUSIVE.1": "No reliable screening result was produced; do not interpret the model score.",
    "rec.any.INCONCLUSIVE.2": "Repeat the image if quality was the problem, or refer for clinician review.",
    "rec.any.REJECTED.1": "The image was not analysed. Follow the suggested corrective action and upload again.",
    "rec.common.no_treatment": "This tool does not recommend medicines or treatment. Management decisions "
                               "belong to the treating clinician.",

    # ── Questionnaire ────────────────────────────────────────────
    "q.header": "Optional symptoms & risk factors (stored separately from the image result)",
    "q.tb.cough_2w": "Cough for 2 weeks or more",
    "q.tb.fever": "Fever",
    "q.tb.night_sweats": "Night sweats",
    "q.tb.weight_loss": "Unexplained weight loss",
    "q.tb.tb_contact": "Previous TB or contact with a TB patient",
    "q.dr.diabetes_type": "Type of diabetes",
    "q.dr.diabetes_years": "Years since diabetes diagnosis",
    "q.dr.eye_complications": "Previous eye complications",
    "q.dr.last_eye_exam": "Last eye examination",
    "q.oral.tobacco": "Tobacco use (smoking or chewing)",
    "q.oral.gutka_paan": "Gutka / paan / areca nut use",
    "q.oral.ulcer_2w": "Mouth ulcer not healing for 2 weeks or more",
    "q.oral.dysphagia": "Difficulty swallowing",
    "opt.type1": "Type 1", "opt.type2": "Type 2", "opt.other": "Other / unknown",
    "opt.lt1y": "Within 12 months", "opt.gt1y": "More than 12 months ago", "opt.never": "Never / unknown",

    # Questionnaire rules (shown with rule IDs)
    "rule.TB-Q1": "Cough for 2 weeks or more — a presumptive TB symptom; clinical evaluation and sputum testing "
                  "are indicated by this symptom alone.",
    "rule.TB-Q2": "Two or more of fever, night sweats and weight loss — clinical evaluation for TB is indicated.",
    "rule.TB-Q3": "Previous TB or TB contact — inform the clinician; contacts may need screening per programme guidelines.",
    "rule.DR-Q1": "Eye examination not done in the last 12 months — a diabetic eye examination is due.",
    "rule.DR-Q2": "Previous eye complications — ophthalmology follow-up is advised.",
    "rule.DR-Q3": "Diabetes for 10 years or more — higher risk of retinopathy; ensure regular eye examinations.",
    "rule.ORAL-Q1": "Mouth ulcer for 2 weeks or more — needs clinical oral examination.",
    "rule.ORAL-Q2": "Difficulty swallowing — needs clinical (ENT / head-and-neck) evaluation.",
    "rule.ORAL-Q3": "Tobacco / gutka / paan use — a known oral cancer risk factor; regular oral examination "
                    "and cessation support are advised.",
    "q.none": "No questionnaire answers recorded.",
    "q.no_rules": "No questionnaire-based referral rules triggered.",

    # ── PDF / labels ─────────────────────────────────────────────
    "pdf.title": "AI-Assisted Screening Report",
    "pdf.subtitle": "Screening aid — not a diagnostic certificate",
    "pdf.patient_ref": "Patient reference",
    "pdf.internal_id": "Internal patient ID",
    "pdf.date": "Screening date (UTC)",
    "pdf.module": "Screening module",
    "pdf.quality": "Image quality",
    "pdf.quality_ok": "Passed automated checks",
    "pdf.result": "Screening result",
    "pdf.score": "Model score",
    "pdf.threshold": "Decision threshold",
    "pdf.model_version": "Model version",
    "pdf.routing": "Modality routing",
    "pdf.gradcam": "Grad-CAM visual explanation",
    "pdf.gradcam_unavailable": "Grad-CAM not available for this report",
    "pdf.recommendations": "Suggested next steps",
    "pdf.questionnaire": "Questionnaire findings (not part of the image model)",
    "pdf.limitations": "Limitations",
    "pdf.limit.1": "The model was trained on a small public dataset and has not been clinically validated.",
    "pdf.limit.2": "The model score is not a calibrated probability of disease.",
    "pdf.limit.3": "A result without a screening flag does not exclude disease.",
    "pdf.limit.4": "Grad-CAM highlights model attention only and is not a lesion map.",
    "pdf.disclaimer": "Disclaimer",
    "pdf.footer": "Generated by MediScan AI (offline). Page",
    "pdf.reviewed_by": "Reviewed by (clinician name & signature)",
    "pdf.processing_time": "Processing time",
}

HI: dict[str, str] = {
    "disclaimer": "यह एक प्रायोगिक एआई स्क्रीनिंग सहायक है, चिकित्सा निदान नहीं। परिणामों की समीक्षा किसी योग्य "
                  "स्वास्थ्य पेशेवर द्वारा की जानी चाहिए और उचित नैदानिक जाँच से उनकी पुष्टि की जानी चाहिए।",
    "simulated_banner": "सिम्युलेटेड आउटपुट — कोई प्रशिक्षित मॉडल नहीं चलाया गया। केवल इंटरफ़ेस अभ्यास के लिए; "
                        "यह स्क्रीनिंग परिणाम नहीं है।",
    "score_note": "मॉडल स्कोर मॉडल का कच्चा आउटपुट है, रोग की अंशांकित (कैलिब्रेटेड) संभावना नहीं।",
    "gradcam_note": "ग्रैड-कैम दिखाता है कि मॉडल ने छवि में कहाँ ध्यान दिया। यह परिणाम के सही होने का प्रमाण नहीं है।",

    "disease.tb": "क्षय रोग / टीबी (छाती का एक्स-रे)",
    "disease.dr": "डायबिटिक रेटिनोपैथी (रेटिना फंडस फोटो)",
    "disease.oral": "मुख कैंसर — OSCC (हिस्टोपैथोलॉजी)",
    "modality.chest_xray": "छाती का एक्स-रे",
    "modality.fundus": "रेटिना फंडस फोटो",
    "modality.histopathology": "मुख हिस्टोपैथोलॉजी",
    "routing.manual_selection": "स्वयं चुना गया (प्रशिक्षित वर्गीकरण मॉडल द्वारा सत्यापित नहीं)",
    "routing.selected_specialist": "उपयुक्तता जाँच के बाद सीधे चुने गए रोग विशेषज्ञ को भेजा गया",
    "routing.trained_gatekeeper": "प्रशिक्षित गेटकीपर मॉडल द्वारा स्वचालित रूप से पहचाना गया",
    "routing.manual_selection+trained_gatekeeper": "स्वयं चुना गया और गेटकीपर मॉडल द्वारा पुष्टि की गई",

    "status.SCREENING_FLAGGED": "स्क्रीनिंग में संकेत मिला — आगे जाँच की सलाह दी जाती है",
    "status.NO_FLAG_DETECTED": "स्क्रीनिंग में कोई संकेत नहीं मिला — इससे रोग की संभावना समाप्त नहीं होती",
    "status.INCONCLUSIVE": "अनिर्णायक — कोई विश्वसनीय स्क्रीनिंग परिणाम नहीं",
    "status.REJECTED": "छवि अस्वीकृत — विश्लेषण नहीं किया गया",

    "reason.WITHIN_UNCERTAINTY_BAND": "मॉडल स्कोर निर्णय सीमा (थ्रेशोल्ड) के बहुत निकट है, इसलिए विश्वसनीय परिणाम नहीं दिया जा सकता।",
    "reason.OUT_OF_DISTRIBUTION": "यह छवि उस प्रकार की छवियों से भिन्न है जिन पर मॉडल को प्रशिक्षित किया गया था।",
    "reason.PREPROCESSING_ERROR": "छवि को मॉडल की आवश्यकता के अनुसार तैयार नहीं किया जा सका।",
    "reason.INFERENCE_ERROR": "छवि का विश्लेषण करते समय मॉडल विफल हो गया।",
    "reason.MODEL_OUTPUT_INVALID": "मॉडल ने अमान्य आउटपुट दिया, इसलिए कोई परिणाम नहीं दिखाया गया।",
    "reason.MODEL_UNAVAILABLE": "इस मॉड्यूल का मॉडल स्थापित नहीं है या सत्यापन में विफल रहा।",

    "reject.IMAGE_CORRUPTED.why": "फ़ाइल को मान्य छवि के रूप में पढ़ा नहीं जा सका (यह क्षतिग्रस्त या अधूरी हो सकती है)।",
    "reject.IMAGE_CORRUPTED.fix": "छवि को फिर से निर्यात या कैप्चर करें और दोबारा अपलोड करें।",
    "reject.IMAGE_UNSUPPORTED_FORMAT.why": "फ़ाइल प्रारूप {fmt} समर्थित नहीं है।",
    "reject.IMAGE_UNSUPPORTED_FORMAT.fix": "PNG या JPEG छवि अपलोड करें (DICOM को पहले PNG में निर्यात करें)।",
    "reject.IMAGE_TOO_LARGE.why": "फ़ाइल {limit_mb} MB की सीमा से बड़ी है या उसमें बहुत अधिक पिक्सेल हैं।",
    "reject.IMAGE_TOO_LARGE.fix": "छवि का आकार घटाएँ या उसे संपीड़ित करें और फिर से प्रयास करें।",
    "reject.IMAGE_TOO_SMALL.why": "छवि {width}×{height} पिक्सेल की है; प्रत्येक भुजा कम से कम {min_side} पिक्सेल होनी चाहिए।",
    "reject.IMAGE_TOO_SMALL.fix": "मूल, पूर्ण-रिज़ॉल्यूशन वाली छवि अपलोड करें।",
    "reject.IMAGE_EXTREME_ASPECT.why": "छवि का आकार असामान्य है (अनुपात {ratio}:1, अधिकतम {max_ratio}:1); "
                                       "यह कटी हुई छवि या स्क्रीनशॉट हो सकती है।",
    "reject.IMAGE_EXTREME_ASPECT.fix": "पूरी, बिना कटी छवि अपलोड करें।",
    "reject.IMAGE_TOO_BLURRY.why": "विश्लेषण के लिए छवि बहुत धुंधली है (स्पष्टता {score}, न्यूनतम {minimum})।",
    "reject.IMAGE_TOO_BLURRY.fix": "कैमरा या स्लाइड को स्थिर रखते हुए छवि को फ़ोकस में दोबारा लें।",
    "reject.IMAGE_OVEREXPOSED.why": "छवि अत्यधिक उजली (ओवरएक्सपोज़्ड) है; महत्वपूर्ण विवरण खो सकते हैं।",
    "reject.IMAGE_OVEREXPOSED.fix": "एक्सपोज़र या चमक कम करके छवि दोबारा लें।",
    "reject.IMAGE_UNDEREXPOSED.why": "छवि इतनी अँधेरी है कि उसमें उपयोगी विवरण नहीं है।",
    "reject.IMAGE_UNDEREXPOSED.fix": "रोशनी या एक्सपोज़र बढ़ाकर छवि दोबारा लें।",
    "reject.DUPLICATE_IMAGE.why": "एक समान फ़ाइल पहले ही स्क्रीन की जा चुकी है ({where}, {when})। समान फ़ाइल होना यह "
                                  "सिद्ध नहीं करता कि छवियाँ एक ही रोगी की हैं।",
    "reject.DUPLICATE_IMAGE.fix": "जाँचें कि आपने सही छवि चुनी है। जानबूझकर दोबारा स्क्रीन करने के लिए "
                                  "'समान फ़ाइल की पुनः स्क्रीनिंग की अनुमति दें' चुनें।",
    "reject.UNSUPPORTED_MODALITY.why": "यह छवि प्रकार समर्थित नहीं है ({label})।",
    "reject.UNSUPPORTED_MODALITY.fix": "छाती का एक्स-रे, रेटिना फंडस फोटो या मुख हिस्टोपैथोलॉजी छवि अपलोड करें।",
    "reject.UNCERTAIN_MODALITY.why": "छवि का प्रकार विश्वसनीय रूप से निर्धारित नहीं हो सका ({reason})।",
    "reject.UNCERTAIN_MODALITY.fix": "स्क्रीनिंग मॉड्यूल स्वयं चुनें, या समर्थित प्रकार की अधिक स्पष्ट छवि अपलोड करें।",
    "reject.MODALITY_MISMATCH.why": "छवि चुने गए प्रकार जैसी नहीं दिखती (अपेक्षित {expected}, पाया गया {observed})।",
    "reject.MODALITY_MISMATCH.fix": "चुना गया मॉड्यूल जाँचें या सही छवि अपलोड करें।",
    "reject.MODEL_UNAVAILABLE.why": "{module} मॉडल स्थापित नहीं है या सत्यापन में विफल रहा।",
    "reject.MODEL_UNAVAILABLE.fix": "प्रशासक से प्रशिक्षित मॉडल आर्टिफैक्ट आयात करने के लिए कहें।",
    "reject.DISEASE_SELECTION_REQUIRED.why": "स्क्रीनिंग से पहले रोग विशेषज्ञ चुनें; स्वचालित रोग रूटिंग बंद है।",
    "reject.DISEASE_SELECTION_REQUIRED.fix": "छवि संबंधित विशेषज्ञ को भेजने के लिए टीबी, डायबिटिक रेटिनोपैथी या ओरल कैंसर चुनें।",

    "priority.ROUTINE": "सामान्य (रूटीन)",
    "priority.PRIORITY": "प्राथमिकता के साथ रेफ़रल",
    "source.image": "छवि मॉडल",
    "source.questionnaire": "प्रश्नावली",

    "rec.tb.SCREENING_FLAGGED.1": "नैदानिक मूल्यांकन के लिए योग्य चिकित्सक (मेडिकल ऑफिसर) के पास भेजें।",
    "rec.tb.SCREENING_FLAGGED.2": "राष्ट्रीय टीबी कार्यक्रम के दिशानिर्देशों के अनुसार बलगम (स्पुटम) जाँच कराएँ, "
                                  "अधिमानतः WHO-अनुशंसित आणविक परीक्षण (NAAT, जैसे CBNAAT/Truenat)।",
    "rec.tb.SCREENING_FLAGGED.3": "छाती के एक्स-रे की समीक्षा रेडियोलॉजिस्ट या चिकित्सक द्वारा की जानी चाहिए।",
    "rec.tb.NO_FLAG_DETECTED.1": "इस छवि पर टीबी से संबंधित कोई स्क्रीनिंग संकेत नहीं मिला। इससे टीबी की संभावना समाप्त नहीं होती।",
    "rec.tb.NO_FLAG_DETECTED.2": "यदि 2 सप्ताह या अधिक से खाँसी, बुखार, रात में पसीना या वज़न घटना हो, तो इस परिणाम "
                                 "के बावजूद नैदानिक मूल्यांकन और बलगम जाँच के लिए भेजें।",
    "rec.dr.SCREENING_FLAGGED.1": "जाँच के लिए नेत्र विशेषज्ञ (ऑफ्थैल्मोलॉजिस्ट) के पास भेजें।",
    "rec.dr.SCREENING_FLAGGED.2": "पुतली फैलाकर (डाइलेटेड) रेटिना की जाँच की सलाह दी जाती है।",
    "rec.dr.SCREENING_FLAGGED.3": "चिकित्सकीय रूप से आवश्यक होने पर ऑप्टिकल कोहेरेंस टोमोग्राफी (OCT), "
                                  "उदाहरण के लिए मैक्युलर एडिमा का संदेह होने पर।",
    "rec.dr.NO_FLAG_DETECTED.1": "इस छवि पर डायबिटिक रेटिनोपैथी का कोई स्क्रीनिंग संकेत नहीं मिला। "
                                 "इससे डायबिटिक रेटिनोपैथी की संभावना समाप्त नहीं होती।",
    "rec.dr.NO_FLAG_DETECTED.2": "उपचार करने वाले चिकित्सक की सलाह के अनुसार नियमित रूप से मधुमेह संबंधी नेत्र जाँच जारी रखें।",
    "rec.oral.SCREENING_FLAGGED.1": "इस स्लाइड की समीक्षा किसी योग्य पैथोलॉजिस्ट द्वारा की जानी चाहिए; "
                                    "यह संकेत हिस्टोपैथोलॉजिकल निदान नहीं है।",
    "rec.oral.SCREENING_FLAGGED.2": "मुख की जाँच और नैदानिक मिलान के लिए विशेषज्ञ (ओरल एवं मैक्सिलोफेशियल सर्जरी, "
                                    "ईएनटी या हेड-एंड-नेक ऑन्कोलॉजी) के पास भेजें।",
    "rec.oral.SCREENING_FLAGGED.3": "आगे बायोप्सी या ऊतक नमूना केवल विशेषज्ञ की सलाह पर ही लें।",
    "rec.oral.NO_FLAG_DETECTED.1": "इस छवि पर OSCC का कोई स्क्रीनिंग संकेत नहीं मिला। इससे मुख कैंसर की संभावना समाप्त "
                                   "नहीं होती; स्लाइड की पैथोलॉजिस्ट द्वारा समीक्षा अब भी आवश्यक है।",
    "rec.oral.NO_FLAG_DETECTED.2": "2 सप्ताह से अधिक समय तक रहने वाले मुँह के छाले, सफ़ेद या लाल धब्बे, या गाँठ होने पर "
                                   "इस परिणाम के बावजूद नैदानिक जाँच आवश्यक है।",
    "rec.any.INCONCLUSIVE.1": "कोई विश्वसनीय स्क्रीनिंग परिणाम नहीं मिला; मॉडल स्कोर की व्याख्या न करें।",
    "rec.any.INCONCLUSIVE.2": "यदि समस्या छवि की गुणवत्ता की थी तो छवि दोबारा लें, अन्यथा चिकित्सक की समीक्षा के लिए भेजें।",
    "rec.any.REJECTED.1": "छवि का विश्लेषण नहीं किया गया। सुझाए गए सुधारात्मक कदम का पालन करें और दोबारा अपलोड करें।",
    "rec.common.no_treatment": "यह उपकरण दवाओं या उपचार की सलाह नहीं देता। उपचार संबंधी निर्णय उपचार करने वाले चिकित्सक के हैं।",

    "q.header": "वैकल्पिक लक्षण और जोखिम कारक (छवि परिणाम से अलग संग्रहीत)",
    "q.tb.cough_2w": "2 सप्ताह या उससे अधिक से खाँसी",
    "q.tb.fever": "बुखार",
    "q.tb.night_sweats": "रात में पसीना",
    "q.tb.weight_loss": "बिना कारण वज़न घटना",
    "q.tb.tb_contact": "पहले टीबी हुई हो या टीबी रोगी के संपर्क में रहे हों",
    "q.dr.diabetes_type": "मधुमेह का प्रकार",
    "q.dr.diabetes_years": "मधुमेह का निदान हुए कितने वर्ष",
    "q.dr.eye_complications": "पहले आँखों से संबंधित जटिलताएँ",
    "q.dr.last_eye_exam": "पिछली नेत्र जाँच",
    "q.oral.tobacco": "तंबाकू का सेवन (धूम्रपान या चबाना)",
    "q.oral.gutka_paan": "गुटखा / पान / सुपारी का सेवन",
    "q.oral.ulcer_2w": "मुँह का छाला जो 2 सप्ताह या अधिक से ठीक नहीं हुआ",
    "q.oral.dysphagia": "निगलने में कठिनाई",
    "opt.type1": "टाइप 1", "opt.type2": "टाइप 2", "opt.other": "अन्य / ज्ञात नहीं",
    "opt.lt1y": "पिछले 12 महीनों में", "opt.gt1y": "12 महीने से पहले", "opt.never": "कभी नहीं / ज्ञात नहीं",

    "rule.TB-Q1": "2 सप्ताह या अधिक से खाँसी — यह टीबी का संभावित लक्षण है; केवल इस लक्षण के आधार पर भी नैदानिक "
                  "मूल्यांकन और बलगम जाँच आवश्यक है।",
    "rule.TB-Q2": "बुखार, रात में पसीना और वज़न घटने में से दो या अधिक — टीबी के लिए नैदानिक मूल्यांकन आवश्यक है।",
    "rule.TB-Q3": "पहले टीबी या टीबी संपर्क — चिकित्सक को सूचित करें; कार्यक्रम दिशानिर्देशों के अनुसार संपर्कों की "
                  "जाँच आवश्यक हो सकती है।",
    "rule.DR-Q1": "पिछले 12 महीनों में नेत्र जाँच नहीं हुई — मधुमेह संबंधी नेत्र जाँच का समय हो गया है।",
    "rule.DR-Q2": "पहले आँखों से संबंधित जटिलताएँ — नेत्र विशेषज्ञ से फ़ॉलो-अप की सलाह दी जाती है।",
    "rule.DR-Q3": "10 वर्ष या अधिक से मधुमेह — रेटिनोपैथी का अधिक जोखिम; नियमित नेत्र जाँच सुनिश्चित करें।",
    "rule.ORAL-Q1": "2 सप्ताह या अधिक से मुँह का छाला — मुख की नैदानिक जाँच आवश्यक है।",
    "rule.ORAL-Q2": "निगलने में कठिनाई — नैदानिक (ईएनटी / हेड-एंड-नेक) मूल्यांकन आवश्यक है।",
    "rule.ORAL-Q3": "तंबाकू / गुटखा / पान का सेवन — मुख कैंसर का ज्ञात जोखिम कारक; नियमित मुख जाँच और सेवन छोड़ने "
                    "में सहायता की सलाह दी जाती है।",
    "q.none": "प्रश्नावली के कोई उत्तर दर्ज नहीं किए गए।",
    "q.no_rules": "प्रश्नावली पर आधारित कोई रेफ़रल नियम लागू नहीं हुआ।",

    "pdf.title": "एआई-सहायित स्क्रीनिंग रिपोर्ट",
    "pdf.subtitle": "स्क्रीनिंग सहायक — नैदानिक प्रमाणपत्र नहीं",
    "pdf.patient_ref": "रोगी संदर्भ",
    "pdf.internal_id": "आंतरिक रोगी आईडी",
    "pdf.date": "स्क्रीनिंग तिथि (UTC)",
    "pdf.module": "स्क्रीनिंग मॉड्यूल",
    "pdf.quality": "छवि गुणवत्ता",
    "pdf.quality_ok": "स्वचालित जाँच में उत्तीर्ण",
    "pdf.result": "स्क्रीनिंग परिणाम",
    "pdf.score": "मॉडल स्कोर",
    "pdf.threshold": "निर्णय सीमा (थ्रेशोल्ड)",
    "pdf.model_version": "मॉडल संस्करण",
    "pdf.routing": "छवि प्रकार निर्धारण",
    "pdf.gradcam": "ग्रैड-कैम दृश्य व्याख्या",
    "pdf.gradcam_unavailable": "इस रिपोर्ट के लिए ग्रैड-कैम उपलब्ध नहीं है",
    "pdf.recommendations": "सुझाए गए अगले कदम",
    "pdf.questionnaire": "प्रश्नावली के निष्कर्ष (छवि मॉडल का भाग नहीं)",
    "pdf.limitations": "सीमाएँ",
    "pdf.limit.1": "मॉडल को एक छोटे सार्वजनिक डेटासेट पर प्रशिक्षित किया गया है और इसका नैदानिक सत्यापन नहीं हुआ है।",
    "pdf.limit.2": "मॉडल स्कोर रोग की अंशांकित संभावना नहीं है।",
    "pdf.limit.3": "स्क्रीनिंग संकेत न मिलने से रोग की संभावना समाप्त नहीं होती।",
    "pdf.limit.4": "ग्रैड-कैम केवल मॉडल का ध्यान दिखाता है; यह घाव का मानचित्र नहीं है।",
    "pdf.disclaimer": "अस्वीकरण",
    "pdf.footer": "MediScan AI (ऑफ़लाइन) द्वारा निर्मित। पृष्ठ",
    "pdf.reviewed_by": "समीक्षक (चिकित्सक का नाम और हस्ताक्षर)",
    "pdf.processing_time": "प्रसंस्करण समय",
}

_TABLES = {"en": EN, "hi": HI}


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def t(key: str, lang: str = "en", **kw) -> str:
    table = _TABLES.get(lang, EN)
    text = table.get(key) or EN.get(key) or key
    if kw:
        try:
            text = string.Formatter().vformat(text, (), _SafeDict(kw))
        except (ValueError, IndexError):
            pass
    return text


def has(key: str, lang: str = "en") -> bool:
    return key in _TABLES.get(lang, {})


def missing_keys(lang: str) -> list[str]:
    return sorted(set(EN) - set(_TABLES[lang]))
