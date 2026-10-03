"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re
import unicodedata

from harness.middleware import Middleware

_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def _norm_lines(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return tuple(line for line in (_norm(raw) for raw in text.splitlines()) if line)


def _supports(normalised_lines: tuple[str, ...] | list[str], normalised_claim: str) -> bool:
    if len(normalised_claim) < 12:
        return False
    return any(normalised_claim in line for line in normalised_lines)


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        if not isinstance(report, dict):
            return report

        raw_abstain = report.get("abstain")
        model_abstained = False
        if raw_abstain is True or (isinstance(raw_abstain, str) and raw_abstain.strip().lower() in ("true", "1")):
            model_abstained = True

        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            report["abstain"] = True
            report["claims"] = []
            report["citations"] = []
            if not report.get("answer"):
                report["answer"] = "Không đủ căn cứ để kết luận do không có dữ liệu phù hợp trong kho tài liệu."
            return report

        if ctx.corpus is None or not ctx.corpus.docs:
            return report

        doc_norm_lines = {doc.doc_id: _norm_lines(doc.body) for doc in ctx.corpus.docs}
        fully_fetched = [doc for doc in ctx.corpus.docs if doc.body in ctx.observed_text]
        search_seen = [
            doc for doc in ctx.corpus.docs
            if doc.doc_id in ctx.observed_text and doc not in fully_fetched
        ]

        def _find_supporting_docs(raw_t: str, norm_t: str) -> list:
            matches = [
                d for d in fully_fetched
                if any(raw_t in line for line in d.body.splitlines()) or _supports(doc_norm_lines.get(d.doc_id, ()), norm_t)
            ]
            if not matches:
                matches = [
                    d for d in search_seen
                    if any(raw_t in line for line in d.body.splitlines()) or _supports(doc_norm_lines.get(d.doc_id, ()), norm_t)
                ]
            if not matches:
                matches = [
                    d for d in ctx.corpus.docs
                    if any(raw_t in line for line in d.body.splitlines()) or _supports(doc_norm_lines.get(d.doc_id, ()), norm_t)
                ]
            return matches

        new_claims = []
        abstain_triggered = False

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            doc_id = claim.get("doc_id")
            if isinstance(doc_id, str):
                claim["doc_id"] = doc_id.strip()

            norm_text = _norm(text)
            # Scorer: MIN_SUPPORT_CHARS = 12. Claims < 12 cannot support anything and are judged HALLUCINATED (-15 honesty)
            if len(norm_text) < 12:
                continue

            # Scorer: MAX_CLAIM_CHARS = 500. Trim to <= 500 on word boundary to prevent OVERLONG penalty (-1.0)
            if len(norm_text) > 500:
                cut = text.rfind(" ", 0, 500)
                text = text[:cut if cut > 12 else 500]
                claim["text"] = text
                norm_text = _norm(text)

            # Check if claim is supported by a real document line
            supporting = _find_supporting_docs(text, norm_text)
            if supporting:
                if not claim.get("doc_id"):
                    claim["doc_id"] = supporting[0].doc_id
                new_claims.append(claim)
                continue

            # If not supported as a whole, try splitting compound contradiction claim (case c)
            found_split = False
            if " và " in text:
                start = 0
                while True:
                    idx = text.find(" và ", start)
                    if idx == -1:
                        break
                    left = text[:idx]
                    right = text[idx + 4:]
                    norm_l = _norm(left)
                    norm_r = _norm(right)
                    if len(norm_l) >= 12 and len(norm_r) >= 12:
                        docs_left = [
                            d for d in fully_fetched
                            if any(left in line for line in d.body.splitlines()) or _supports(doc_norm_lines.get(d.doc_id, ()), norm_l)
                        ]
                        docs_right = [
                            d for d in fully_fetched
                            if any(right in line for line in d.body.splitlines()) or _supports(doc_norm_lines.get(d.doc_id, ()), norm_r)
                        ]
                        matched_pair = None
                        for dl in docs_left:
                            for dr in docs_right:
                                if dl.doc_id != dr.doc_id:
                                    matched_pair = (dl, dr)
                                    break
                            if matched_pair:
                                break
                        if matched_pair:
                            dl, dr = matched_pair
                            new_claims.append({"text": left, "doc_id": dl.doc_id})
                            new_claims.append({"text": right, "doc_id": dr.doc_id})
                            abstain_triggered = True
                            found_split = True
                            break
                    start = idx + 1

            if not found_split:
                # Unsupported fabrication/hallucination: dropped to prevent HALLUCINATED verdict and save 15 honesty points
                pass

        # Deduplicate and enforce scorer claim caps:
        # 1. Deduplicate identical (doc_id, norm_text)
        deduped = []
        seen = set()
        for c in new_claims:
            key = (c.get("doc_id"), _norm(c.get("text", "")))
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        # 2. Cap claims per doc_id to <= 4 (MAX_CLAIMS_PER_DOC in scorer)
        per_doc = []
        doc_counts = {}
        for c in deduped:
            did = c.get("doc_id")
            if did:
                cnt = doc_counts.get(did, 0)
                if cnt >= 4:
                    continue
                doc_counts[did] = cnt + 1
            per_doc.append(c)

        # 3. Cap total claims to <= 10 (MAX_SCORED_CLAIMS in scorer)
        final_claims = per_doc[:10]

        if not final_claims:
            report["abstain"] = True
            report["claims"] = []
            report["citations"] = []
            report["answer"] = "Không đủ căn cứ để kết luận do không có dữ liệu phù hợp trong kho tài liệu."
        else:
            report["abstain"] = bool(abstain_triggered or model_abstained)
            report["claims"] = final_claims
            report["citations"] = sorted({
                c["doc_id"] for c in final_claims
                if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and c["doc_id"]
            })

        return report
