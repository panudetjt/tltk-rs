/**
 * app.js - Main UI controller for TLTK Web Toolbox
 */

import { selectFlavor, probeSimd } from './probe.js';
import { TltkManager, TIER2_FUNCTIONS, TIER1_FILES, TIER2_FILES, assetUrl } from './loader.js';

// Metadata and Thai descriptions for TLTK functions
export const FUNCTIONS = [
  {
    id: 'word_segment',
    name: 'word_segment',
    thaiName: 'ตัดคำ (Word Segmentation)',
    desc: 'ตัดข้อความภาษาไทยออกเป็นคำ คั่นด้วยสัญลักษณ์ | และ <s/> ที่ท้ายประโยค',
    placeholder: 'แมวกินปลา',
    tier: 1
  },
  {
    id: 'syl_segment',
    name: 'syl_segment',
    thaiName: 'ตัดพยางค์ (Syllable Segmentation)',
    desc: 'ตัดคำหรือข้อความออกเป็นพยางค์ย่อย คั่นด้วยสัญลักษณ์ ~',
    placeholder: 'มหาวิทยาลัย',
    tier: 1
  },
  {
    id: 'segment',
    name: 'segment',
    thaiName: 'ตัดคำและพยางค์พร้อมระบุโครงสร้าง (Segment with Tags)',
    desc: 'ตัดข้อความระดับคำและพยางค์ พร้อมระบุประเภทคำผสมและแท็กโครงสร้างส่วนย่อย',
    placeholder: 'อาหารอร่อย',
    tier: 2
  },
  {
    id: 'th2ipa',
    name: 'th2ipa',
    thaiName: 'แปลงตัวเขียนเป็นสัทอักษรสากล (Thai to IPA)',
    desc: 'แปลงข้อความภาษาไทยเป็นสัทอักษรสากล (International Phonetic Alphabet) พร้อมระดับวรรณยุกต์',
    placeholder: 'กรุงเทพมหานคร',
    tier: 1
  },
  {
    id: 'th2roman',
    name: 'th2roman',
    thaiName: 'ถอดอักษรไทยเป็นอักษรโรมัน (Romanization)',
    desc: 'ถอดอักษรไทยเป็นอักษรโรมันตามหลักเกณฑ์การถอดเสียง',
    placeholder: 'สุราษฎร์ธานี',
    tier: 1
  },
  {
    id: 'g2p',
    name: 'g2p',
    thaiName: 'แปลงตัวเขียนเป็นเสียงอ่านสัทศาสตร์ (Grapheme-to-Phoneme)',
    desc: 'แปลงตัวเขียนอักษรไทยเป็นรหัสหน่วยเสียงภาษาไทย',
    placeholder: 'สวัสดีครับ',
    tier: 1
  },
  {
    id: 'th2read',
    name: 'th2read',
    thaiName: 'แปลงตัวเขียนเป็นคำอ่านภาษาไทย (Thai Reading)',
    desc: 'แปลงคำเขียนภาษาไทยเป็นคำอ่านภาษาไทยตามเสียงพยางค์',
    placeholder: 'สุราษฎร์ธานี',
    tier: 1
  },
  {
    id: 'spell_variants',
    name: 'spell_variants',
    thaiName: 'ค้นหาคำสะกดใกล้เคียง (Spelling Variants)',
    desc: 'ค้นหารูปสะกดที่เป็นไปได้หรือคำพ้องเสียงในคลังคำ',
    placeholder: 'แมวกินปลา',
    tier: 1
  },
  {
    id: 'spoonerism',
    name: 'spoonerism',
    thaiName: 'คำผวน (Spoonerism)',
    desc: 'สลับเสียงสระและพยัญชนะท้ายระหว่างพยางค์เพื่อสร้างคำผวน',
    placeholder: 'ควายตาม',
    tier: 1
  },
  {
    id: 'pos_tag',
    name: 'pos_tag',
    thaiName: 'ระบุชนิดของคำ (Part-of-Speech Tagging)',
    desc: 'กำกับหน้าที่และประเภทของคำในประโยค (POS) เช่น นาม กริยา คุณศัพท์',
    placeholder: 'แมวกินปลา',
    tier: 1
  },
  {
    id: 'ner_tag',
    name: 'ner_tag',
    thaiName: 'ระบุชื่อเฉพาะ (Named Entity Recognition)',
    desc: 'ระบุชื่อเฉพาะ เช่น บุคคล (Person) องค์กร (Organization) สถานที่ (Location)',
    placeholder: 'นายสมชายเดินทางไปกรุงเทพมหานคร',
    tier: 1
  },
  {
    id: 'similar_words',
    name: 'similar_words',
    thaiName: 'ค้นหาคำที่มีความหมายใกล้เคียง (Similar Words)',
    desc: 'ค้นหาคำที่มีเวกเตอร์ความหมายใกล้เคียงจากโมเดล Word2Vec ตามเกณฑ์ cutoff ที่ระบุ',
    placeholder: 'แมว',
    tier: 2,
    hasCutoff: true
  },
  {
    id: 'similarity',
    name: 'similarity',
    thaiName: 'วัดความคล้ายระหว่างสองคำ (Word Similarity)',
    desc: 'คำนวณค่าความคล้ายคลึงของเวกเตอร์ความหมายระหว่าง 2 คำ (คั่นด้วยเว้นวรรค)',
    placeholder: 'แมว สุนัข',
    tier: 2
  },
  {
    id: 'outofgroup',
    name: 'outofgroup',
    thaiName: 'หาคำที่ไม่เข้าพวก (Out of Group)',
    desc: 'ระบุคำที่มีความหมายไม่เข้าพวกจากกลุ่มคำ (ใส่คำตั้งแต่ 3 คำขึ้นไป คั่นด้วยเว้นวรรค)',
    placeholder: 'แมว สุนัข ปลา วัว โต๊ะ',
    tier: 2
  },
  {
    id: 'analogy',
    name: 'analogy',
    thaiName: 'เปรียบเทียบความสัมพันธ์ของคำ (Word Analogy)',
    desc: 'คำนวณความสัมพันธ์ของคู่คำ A:B :: C:? (ใส่คำ 3 คำ คั่นด้วยเว้นวรรค เช่น: พ่อ แม่ ลูกชาย)',
    placeholder: 'พ่อ แม่ ลูกชาย',
    tier: 2
  },
  {
    id: 'plotwords',
    name: 'plotwords',
    thaiName: 'พล็อตแผนภาพเวกเตอร์คำ 2 มิติ (Plot Words PCA)',
    desc: 'แสดงภาพกราฟการกระจายตัวของคำในระนาบ 2 มิติ PCA จากโมเดล Word2Vec',
    placeholder: 'แมว สุนัข เสือ ปลานิล ช้าง ม้า',
    tier: 2
  },
  {
    id: 'w2v_diffplot',
    name: 'w2v_diffplot',
    thaiName: 'พล็อตส่วนต่างเวกเตอร์คำ (Word Vector Difference Plot)',
    desc: 'แสดงกราฟเส้นส่วนต่างของเวกเตอร์ระหว่างคำหลักกับคำเปรียบเทียบ 2 คู่ (ใส่ 3 คำคั่นด้วยเว้นวรรค: คำหลัก คำเทียบ 1 คำเทียบ 2)',
    placeholder: 'แมว ปลา นก',
    tier: 2
  },
  {
    id: 'w2v_dimplot',
    name: 'w2v_dimplot',
    thaiName: 'พล็อตเวกเตอร์รายมิติ (Word Vector Dimension Plot)',
    desc: 'แสดงกราฟเส้นค่าเวกเตอร์ 100 มิติของคำแต่ละคำจากโมเดล Word2Vec',
    placeholder: 'แมว ปลา นก',
    tier: 2
  },
  {
    id: 'w2v_compare_color',
    name: 'w2v_compare_color',
    thaiName: 'เปรียบเทียบเวกเตอร์คำแบบสี (Word Vector Color Comparison)',
    desc: 'แสดงแผนภาพจุดสีแบบ Spectral เปรียบเทียบค่าเวกเตอร์ 100 มิติของหลายคำ',
    placeholder: 'แมว ปลา นก',
    tier: 2
  },
  {
    id: 'textana',
    name: 'textana',
    thaiName: 'วิเคราะห์คุณลักษณะข้อความ (Text Analysis)',
    desc: 'คำนวณคุณลักษณะระดับข้อความ เช่น จำนวนคำ/พยางค์ ความหลากหลายของคำ (LdvTTR) ระดับคำศัพท์ TNC/TTC และความถูกต้องการอ่านออกเสียง (ใส่หลายย่อหน้าคั่นด้วยบรรทัดใหม่) ผลลัพธ์เป็น JSON',
    placeholder: 'แมวกินปลาอยู่ในครัว\nหมูกินข้าวในถัง',
    tier: 1
  },
  {
    id: 'chunk',
    name: 'chunk',
    thaiName: 'แบ่งหน่วยความหมายและวิเคราะห์โครงสร้าง (Chunk Parsing)',
    desc: 'แบ่งหน่วยข้อความระดับประโยคย่อย (EDU) พร้อมตัดคำ กำกับชนิดของคำ และระบุชื่อเฉพาะ คั่นด้วยแท็ก <u/>',
    placeholder: 'นายสมชาย ใจดี เดินทางไปทำงานที่กรุงเทพฯ',
    tier: 2
  },
  {
    id: 'tnc_tag',
    name: 'tnc_tag',
    thaiName: 'กำกับเสียงอ่านและชนิดคำแบบ TNC (TNC Tagging)',
    desc: 'กำกับเสียงอ่านสัทอักษรและชนิดของคำในรูปแบบ XML สำหรับคลังข้อมูล TNC',
    placeholder: 'แมวกินปลา',
    tier: 2
    id: 'th2ipa_all',
    name: 'th2ipa_all',
    thaiName: 'แปลงตัวเขียนเป็นสัทอักษรสากลทุกรูปแบบ (Thai to IPA All Candidates)',
    desc: 'แสดงรูปการแปลงสัทอักษรสากล (IPA) ทุกความเป็นไปได้ของการตัดพยางค์และการออกเสียง',
    placeholder: 'แมวกินปลา',
    tier: 1
  },
  {
    id: 'sylparse_all',
    name: 'sylparse_all',
    thaiName: 'แจกแจงโครงสร้างพยางค์ทุกรูปแบบ (Syllable Parse All Candidates)',
    desc: 'วิเคราะห์โครงสร้างพยางค์และการอ่านออกเสียงทุกรูปแบบที่เป็นไปได้ของคำหรือข้อความ',
    placeholder: 'แมว',
    tier: 1
  }
,
  {
    id: 'wrd_len',
    name: 'wrd_len',
    thaiName: 'วัดขนาดความยาวคำ (Word Length Class)',
    desc: 'จำแนกขนาดความยาวของคำเป็น s (<=10), m (>10 และ <=20), หรือ l (>20)',
    placeholder: 'มหาวิทยาลัย',
    tier: 1
  },
  {
    id: 'match_engtext',
    name: 'match_engtext',
    thaiName: 'ตรวจสอบข้อความภาษาอังกฤษ/สัญลักษณ์ (Match English Text)',
    desc: 'ตรวจสอบว่าข้อความประกอบด้วยตัวอักษรภาษาอังกฤษ ตัวเลข และสัญลักษณ์ ASCII ทั้งหมดหรือไม่',
    placeholder: 'hello 123',
    tier: 1
  },
  {
    id: 'known',
    name: 'known',
    thaiName: 'ค้นหาคำที่มีในพจนานุกรม (Known Words)',
    desc: 'กรองคำที่พบในพจนานุกรมภาษาไทย (ใส่คำคั่นด้วยเว้นวรรค)',
    placeholder: 'แมว หมา คำไม่มีในดิก',
    tier: 1
  },
  {
    id: 'edits2',
    name: 'edits2',
    thaiName: 'สร้างคำแปรผันระยะห่าง 2 (Edits Distance 2)',
    desc: 'สร้างรายการคำแปรผันจากการลบ สลับ แทนที่ และแทรกตัวอักษร 2 ครั้ง (Norvig edits2)',
    placeholder: 'กก',
    tier: 1
  },
  {
    id: 'check_thaidict',
    name: 'check_thaidict',
    thaiName: 'ตรวจสอบคำในพจนานุกรม (Check Thaidict)',
    desc: 'ตรวจสอบว่าคำที่ระบุมีอยู่ในพจนานุกรมไทยหรือไม่ คืนค่า 1 (พบ) หรือ 0 (ไม่พบ)',
    placeholder: 'แมว',
    tier: 1
  },
  {
    id: 'reset_thaidict',
    name: 'reset_thaidict',
    thaiName: 'ล้างพจนานุกรม (Reset Thaidict)',
    desc: 'ล้างข้อมูลคำศัพท์ทั้งหมดในพจนานุกรมที่โหลดอยู่',
    placeholder: '',
    tier: 1
  },
  {
    id: 'read_thaidict',
    name: 'read_thaidict',
    thaiName: 'เพิ่มคำลงในพจนานุกรม (Read Thaidict)',
    desc: 'อ่านและเพิ่มคำศัพท์เข้าสู่พจนานุกรม (ใส่คำบรรทัดละ 1 คำ)',
    placeholder: 'คำใหม่1\nคำใหม่2',
    tier: 1
  }
,
  {
    id: 'dg_print',
    name: 'dg_print',
    thaiName: 'พิมพ์ความสัมพันธ์การพึ่งพา (Dependency Relations)',
    desc: 'แสดงรายการความสัมพันธ์ (deprel, word/tag-i, head/tag-j) จากผล parse แบบ dependency (วาง JSON dict ของ dg_save)',
    placeholder: '{"sentence":"...","words":[{"id":0,"form":"root","pos":"TOP","head":null,"deprel":null}]}',
    tier: 1
  },
  {
    id: 'dg_save',
    name: 'dg_save',
    thaiName: 'บันทึกผล parse เป็น dict (Save Dependency Graph)',
    desc: 'จัดรูปแบบผล dependency parse เป็น {"sentence","words":[{"id","form","pos","head","deprel"}]}',
    placeholder: '{"sentence":"...","words":[{"id":1,"form":"กิน","pos":"VERB","head":0,"deprel":"root"}]}',
    tier: 1
  },
  {
    id: 'print_dtree',
    name: 'print_dtree',
    thaiName: 'พิมพ์ต้นไม้การพึ่งพา (Print Dependency Tree)',
    desc: 'แสดงโครงสร้างต้นไม้ dependency ทีละบรรทัด เรียงตาม id',
    placeholder: '{"sentence":"...","words":[{"id":0,"form":"root","pos":"TOP","head":null,"deprel":null},{"id":1,"form":"กิน","pos":"VERB","head":0,"deprel":"root"}]}',
    tier: 1
  },
  {
    id: 'scan_dtree',
    name: 'scan_dtree',
    thaiName: 'สแกนต้นไม้การพึ่งพา (Scan Dependency Tree)',
    desc: 'สแกนโครงสร้างต้นไม้ dependency ผลลัพธ์เหมือน print_dtree',
    placeholder: '{"sentence":"...","words":[{"id":0,"form":"root","pos":"TOP","head":null,"deprel":null},{"id":1,"form":"กิน","pos":"VERB","head":0,"deprel":"root"}]}',
    tier: 1
  },
  {
    id: 'get_tree_depth',
    name: 'get_tree_depth',
    thaiName: 'ความลึกของคำ (Tree Depth at Word)',
    desc: 'คืนความลึกของต้นไม้ย่อยที่เริ่มจากคำที่ระบุ (ใส่ id ของคำในช่อง cutoff, ค่าเริ่มต้น 0 = ราก)',
    placeholder: '{"sentence":"...","words":[{"id":0,"form":"root","pos":"TOP","head":null,"deprel":null},{"id":1,"form":"กิน","pos":"VERB","head":0,"deprel":"root"}]}',
    tier: 1
  },
  {
    id: 'tree_max_depth',
    name: 'tree_max_depth',
    thaiName: 'ความลึกสูงสุดของต้นไม้ (Maximum Tree Depth)',
    desc: 'คืนความลึกสูงสุดของต้นไม้ dependency ทั้งต้น',
    placeholder: '{"sentence":"...","words":[{"id":0,"form":"root","pos":"TOP","head":null,"deprel":null},{"id":1,"form":"กิน","pos":"VERB","head":0,"deprel":"root"}]}',
    tier: 1
  }
  },
  {
    id: 'w2v',
    name: 'w2v',
    thaiName: 'ดึงเวกเตอร์คำจากโมเดล Word2Vec (Raw Vector)',
    desc: 'แสดงเวกเตอร์ดิบขนาด 100 มิติของคำจากโมเดล Word2Vec (TNC)',
    placeholder: 'แมว',
    tier: 2
  },
  {
    id: 'compound',
    name: 'compound',
    thaiName: 'ตรวจสอบคำประสม (Compound Word)',
    desc: 'ตรวจสอบว่าคำสองคำ (คั่นด้วยเว้นวรรค) รวมกันเป็นคำประสมในโมเดล Word2Vec หรือไม่ พร้อมค่าความคล้าย 3 คู่',
    placeholder: 'ราช การ',
    tier: 2
  },
];

// Map for quick lookup by ID
const FUNCTION_MAP = new Map(FUNCTIONS.map(f => [f.id, f]));

// Single shared TltkManager instance
const manager = new TltkManager();

// DOM Elements
const functionSelect = document.getElementById('functionSelect');
const functionDesc = document.getElementById('functionDesc');
const tierTag = document.getElementById('tierTag');
const cutoffGroup = document.getElementById('cutoffGroup');
const cutoffSelect = document.getElementById('cutoffSelect');
const inputText = document.getElementById('inputText');
const exampleBtn = document.getElementById('exampleBtn');
const clearBtn = document.getElementById('clearBtn');
const runBtn = document.getElementById('runBtn');
const runSpinner = document.getElementById('runSpinner');
const execStats = document.getElementById('execStats');
const progressSection = document.getElementById('progressSection');
const progressMessage = document.getElementById('progressMessage');
const progressPercent = document.getElementById('progressPercent');
const progressBar = document.getElementById('progressBar');
const progressDetail = document.getElementById('progressDetail');
const tier2Warning = document.getElementById('tier2Warning');
const errorPane = document.getElementById('errorPane');
const errorMessage = document.getElementById('errorMessage');
const textOutputWrapper = document.getElementById('textOutputWrapper');
const outputText = document.getElementById('outputText');
const copyBtn = document.getElementById('copyBtn');
const imageOutputWrapper = document.getElementById('imageOutputWrapper');
const outputImage = document.getElementById('outputImage');
const downloadImgLink = document.getElementById('downloadImgLink');

/**
 * Initializes the function dropdown.
 */
function initFunctionSelect() {
  functionSelect.innerHTML = '';

  for (const fn of FUNCTIONS) {
    const opt = document.createElement('option');
    opt.value = fn.id;
    opt.textContent = `${fn.name} — ${fn.thaiName}`;
    functionSelect.appendChild(opt);
  }

  functionSelect.value = 'word_segment';
  onFunctionChange();
}

/**
 * Handles function dropdown change.
 */
function onFunctionChange() {
  const fnId = functionSelect.value;
  const fn = FUNCTION_MAP.get(fnId);
  if (!fn) return;

  functionDesc.textContent = fn.desc;

  if (fn.hasCutoff) {
    cutoffGroup.classList.remove('hidden');
  } else {
    cutoffGroup.classList.add('hidden');
  }

  if (fn.tier === 2) {
    tierTag.textContent = 'Tier 2 (+17.8MB)';
    tierTag.className = 'tier-tag tier-2';
  } else {
    tierTag.textContent = 'Tier 1 (5.4MB)';
    tierTag.className = 'tier-tag';
  }

  // Update placeholder
  inputText.placeholder = fn.placeholder;
}

/**
 * Handles model download progress.
 */
function handleProgress({ phase, fileIndex, totalFiles, fileName, percent }) {
  progressSection.classList.remove('hidden');
  const phaseLabel = phase === 'tier2' ? 'โมเดลขั้นสูง Tier 2' : 'โมเดลพื้นฐาน Tier 1';
  progressMessage.textContent = `กำลังดาวน์โหลด ${phaseLabel} (${fileIndex}/${totalFiles})...`;
  progressPercent.textContent = `${percent}%`;
  progressBar.style.width = `${percent}%`;

  if (fileName && fileName !== 'complete') {
    progressDetail.textContent = `กำลังโหลด: ${fileName}`;
  } else if (fileName === 'complete') {
    progressDetail.textContent = 'ดาวน์โหลดเสร็จสมบูรณ์ กำลังประมวลผล...';
  }
}

/**
 * Runs the selected function with input text.
 */
async function runProcess() {
  const fnId = functionSelect.value;
  const text = inputText.value.trim();
  const cutoff = cutoffSelect ? cutoffSelect.value : 'medium';

  if (!text) {
    showError('กรุณาใส่ข้อความนำเข้าก่อนกดประมวลผล');
    return;
  }

  // Clear previous errors
  errorPane.classList.add('hidden');
  errorMessage.textContent = '';
  runBtn.disabled = true;
  runSpinner.classList.remove('hidden');
  execStats.textContent = 'กำลังประมวลผล...';

  const startTime = performance.now();

  try {
    const result = await manager.process(fnId, text, cutoff, handleProgress);
    const elapsed = Math.round(performance.now() - startTime);

    // Hide progress bar on successful complete
    progressSection.classList.add('hidden');

    execStats.textContent = `ประมวลผลสำเร็จใน ${elapsed} ms`;

    // Render result based on return shape
    if (typeof result === 'string' && result.startsWith('IMAGE:')) {
      const b64 = result.slice(6);
      outputImage.src = `data:image/png;base64,${b64}`;
      downloadImgLink.href = `data:image/png;base64,${b64}`;
      imageOutputWrapper.classList.remove('hidden');
      textOutputWrapper.classList.add('hidden');
    } else {
      // textana returns compact JSON — pretty-print for display only
      let shown = result;
      if (fnId === 'textana' && result) {
        try {
          shown = JSON.stringify(JSON.parse(result), null, 2);
        } catch (_) { /* fall through to raw */ }
      }
      outputText.textContent = shown || '(ผลลัพธ์ว่างเปล่า)';
      textOutputWrapper.classList.remove('hidden');
      imageOutputWrapper.classList.add('hidden');
    }
  } catch (err) {
    progressSection.classList.add('hidden');

    // Check if failure was caused by Tier 2
    if (TIER2_FUNCTIONS.has(fnId) && manager.tier2Error) {
      tier2Warning.classList.remove('hidden');
    }

    showError(err.message || String(err));
    execStats.textContent = 'เกิดข้อผิดพลาด';
  } finally {
    runBtn.disabled = false;
    runSpinner.classList.add('hidden');
  }
}

/**
 * Displays an error message in the error panel.
 */
function showError(msg) {
  errorMessage.textContent = msg;
  errorPane.classList.remove('hidden');
}

/**
 * Copies output text to clipboard.
 */
async function copyOutput() {
  const text = outputText.textContent;
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    const originalText = copyBtn.textContent;
    copyBtn.textContent = 'คัดลอกแล้ว!';
    setTimeout(() => {
      copyBtn.textContent = originalText;
    }, 1500);
  } catch {
    // Fallback selection copy
    const range = document.createRange();
    range.selectNodeContents(outputText);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
}

// Event Listeners
functionSelect.addEventListener('change', onFunctionChange);

exampleBtn.addEventListener('click', () => {
  const fnId = functionSelect.value;
  const fn = FUNCTION_MAP.get(fnId);
  if (fn && fn.placeholder) {
    inputText.value = fn.placeholder;
  }
});

clearBtn.addEventListener('click', () => {
  inputText.value = '';
  inputText.focus();
});

runBtn.addEventListener('click', runProcess);

copyBtn.addEventListener('click', copyOutput);

// Ctrl+Enter or Cmd+Enter to run
inputText.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    runProcess();
  }
});

// Initial Setup
initFunctionSelect();


// ---- PWA: service worker + background asset prefetch (tier 1 -> wasm -> tier 2)
// Fills the SW cache after first paint so the toolbox works offline and the
// first real query is instant. Never blocks the UI; failures are silent.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
    const prefetch = async () => {
      const sw = await navigator.serviceWorker.ready.catch(() => null);
      if (!sw) return;
      const quiet = (u) => fetch(u, { cache: 'force-cache' }).catch(() => {});
      for (const f of TIER1_FILES) await quiet(assetUrl(f));
      const flavor = selectFlavor();
      await quiet(`wasm/${flavor}/tltk_web.js`);
      await quiet(`wasm/${flavor}/tltk_web.wasm`);
      for (const f of TIER2_FILES) await quiet(assetUrl(f));

      // Assets are warm — now secretly BUILD the engine (wasm init + model
      // decode, tier 1 then tier 2). ensureTier1/2 cache their promises, so a
      // user query racing this simply awaits the same work: first click feels
      // instant. All failures stay silent (the query path rethrows properly).
      await manager.ensureTier1().catch(() => {});
      await manager.ensureTier2().catch(() => {});
    };
    window.requestIdleCallback
      ? window.requestIdleCallback(() => { prefetch(); })
      : setTimeout(prefetch, 2000);
  });
}
