#!/usr/bin/env python3
"""Generate the verifier corpus (JSONL): {function, text, cutoff}."""
import json
import sys

words = [
    "แมว", "กิน", "ปลา", "แมวกินปลา", "สวัสดีครับ", "ภาษาไทย", "กรุงเทพมหานคร",
    "ประเทศไทย", "การเดินทาง", "อาหารอร่อย", "คอมพิวเตอร์", "มหาวิทยาลัย",
    "เอกสารประกอบการประชุม", "สุราษฎร์ธานี", "ทรงผม", "ทราบ", "กรุณา", "กฤษณา",
    "ฤดูฝน", "กลัว", "กลางคืน", "แสงดี", "ควายตาม", "มะลิซ้อน", "เรียวเธอคนดอ",
    "อยากกินข้าว", "หนึ่งสองสาม", "๑๒๓๔", "๑๙ ตุลา", "๐๑๐๒",
]
sentences = [
    "แมวกินปลาทะเล",
    "ผมชอบกินข้าวผัดกะเพราหมูสับ",
    "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
    "วันนี้อากาศดีมาก ๆ",
    "เขาไปตลาดซื้อผัก ผลไม้ และของชำ",
    "การประชุมวิชาการจัดขึ้นเมื่อวันที่ 18 มิถุนายน 2566",
    "นายสมชายซื้อของที่จังหวัดเชียงใหม่",
    "บริษัทไทยพาณิชย์จำกัด (มหาชน) เปิดสำนักงานใหม่",
    "I love Thai food มาก",
    "COVID-19 ระบาดในปี 2563",
    "กินข้าวหรือยัง?",
    "ราคา 199.- ถูกมาก!",
    "อ่านหนังสือ...เฉย ๆ",
    "เอบีเอ็น แอมพีเอ็ม",
    "น.ส.สริตา ดอนม่วง ไป สน.ลุมพินี",
    "เมื่อวานนี้ฝนตกหนักมากจนน้ำท่วมถนนหลายสายทำให้การจราจรติดขัดเป็นอย่างมาก",
    "ที่ดักไขมันเป็นอุปกรณ์ที่ใช้ในการแยกไขมันออกจากน้ำเสียก่อนที่จะระบายต่อไป",
]
w2v_words = ["กิน", "แมว", "หมา", "นก", "รถยนต์", "พ่อ แม่ ลูกชาย", "กรุงเทพ", "ข้าว"]

rows = []
text_fn = [
    ("word_segment", words + sentences),
    ("syl_segment", words + sentences),
    ("g2p", words[:12] + sentences[:8]),
    ("th2ipa", words[:12] + sentences[:6]),
    ("th2roman", words[:12] + sentences[:6]),
    ("th2read", words[:10] + sentences[:5]),
    ("spell_variants", ["แมวกินปลา", "มะลิซ้อน", "สุราษฎร์ธานี", "กรุงเทพ", "ทรงผม", "กฤษณา", "สวัสดี", "ครับ", "ขอบคุณ"]),
    ("spoonerism", ["แสงดี", "ควายตาม", "มะลิซ้อน", "ไฟเราแงน", "แมวกินปลา", "สวัสดี", "ประเทศไทย"]),
    ("pos_tag", sentences[:10] + ["แมวกินปลา", "ครับ"]),
    ("ner_tag", sentences[:8]),
    ("segment", sentences[:5]),
]
for fn, texts in text_fn:
    for t in texts:
        rows.append({"function": fn, "text": t})
for w in ["กิน", "แมว", "กรุงเทพ"]:
    for c in ["low", "medium", "high"]:
        rows.append({"function": "similar_words", "text": w, "cutoff": c})
for pair in ["แมว หมา", "กิน ข้าว", "กรุงเทพ เชียงใหม่", "กิน กิน", "แมว แมว"]:
    rows.append({"function": "similarity", "text": pair})
for group in ["แมว หมา นก รถยนต์", "กิน ดื่ม นอน เดิน", "ข้าว น้ำ ผัก คอมพิวเตอร์"]:
    rows.append({"function": "outofgroup", "text": group})
for tri in ["พ่อ แม่ ลูกชาย", "กิน ดื่ม นอน", "กรุงเทพ เชียงใหม่ ขอนแก่น"]:
    rows.append({"function": "analogy", "text": tri})
for w in ["กิน", "แมว หมา", "ข้าว น้ำ ผัก"]:
    rows.append({"function": "plotwords", "text": w})

with open(sys.argv[1] if len(sys.argv) > 1 else "corpus/corpus.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} cases")
