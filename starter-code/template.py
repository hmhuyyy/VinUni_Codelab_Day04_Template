"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import unicodedata
from typing import Dict, Any, List
from tools import TOOL_MAP

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: tư vấn sản phẩm, dịch vụ và tiếp nhận yêu cầu hỗ trợ của Vingroup
- Giọng điệu: chuyên nghiệp, rõ ràng, thân thiện

## AVAILABLE TOOLS
- search_product_catalog(category, max_price): tra cứu danh mục sản phẩm theo loại và giá tối đa
- submit_support_ticket(customer_name, issue_description, priority): ghi nhận yêu cầu hỗ trợ

## CORE RULES
1. Không bịa thông tin, giá, chính sách hoặc trạng thái yêu cầu.
2. Dùng tool tương ứng khi cần tra cứu sản phẩm hoặc tạo yêu cầu hỗ trợ.
3. Với FAQ có dữ liệu xác thực trong ngữ cảnh, trả lời trực tiếp; nếu thiếu dữ liệu, hãy nói rõ.
4. Có thể gọi nhiều tool khi yêu cầu cần cả tra cứu lẫn hỗ trợ.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ sản phẩm và dịch vụ thuộc Vingroup, VinFast, Vinpearl.
- Với nội dung ngoài phạm vi, lịch sự nêu giới hạn hỗ trợ.

## OUTPUT CONTRACT
- Thought: xác định ngắn gọn nhu cầu, không trình bày suy luận nội bộ.
- Action: tên tool và tham số cần gọi, nếu có.
- Observation: kết quả thực tế từ tool.
- Final Answer: trả lời rõ ràng bằng tiếng Việt, dựa trên Observation.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = [{"step": "init", "user_input": user_input}]
        normalized = _normalize_text(user_input)

        is_faq = "bao hanh" in normalized or "chinh sach" in normalized
        product_terms = ("xe", "vinfast", "vinpearl", "du lich", "resort", "khach san")
        catalog_actions = ("xem", "tim", "mua", "gia", "so sanh", "danh sach")
        needs_catalog = (
            not is_faq
            and any(term in normalized for term in product_terms)
            and (
                any(action in normalized for action in catalog_actions)
                or bool(re.search(r"\bco\b.*\bnao\b", normalized))
            )
        )
        ticket_terms = (
            "ho tro", "bi loi", "gap", "khieu nai", "phan hoi", "ticket",
            "lien he", "dat phong", "am moc", "ghi nhan", "van de"
        )
        needs_ticket = any(term in normalized for term in ticket_terms)
        catalog_args = _parse_catalog_args(normalized) if needs_catalog else None
        ticket_args = _parse_ticket_args(user_input, normalized) if needs_ticket else None

        self.trace.append({
            "step": "intent_detection",
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        })

        if is_faq and not needs_catalog and not needs_ticket:
            answer = (
                "Theo thông tin hiện có trong danh mục, pin VinFast VF 5 Plus "
                "được bảo hành 10 năm. Thời hạn này được nêu cho mẫu VF 5 Plus."
            )
            self.trace.append({"step": "final_answer", "answer": answer})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        pending_tools = []
        if catalog_args is not None:
            pending_tools.append(("search_product_catalog", catalog_args))
        if ticket_args is not None:
            pending_tools.append(("submit_support_ticket", ticket_args))

        observations: Dict[str, Any] = {}
        iteration = 0
        while pending_tools and iteration < self.max_iterations:
            tool_name, arguments = pending_tools.pop(0)
            iteration += 1
            result = TOOL_MAP[tool_name](**arguments)
            observations[tool_name] = result
            self.trace.append({
                "step": "tool_call",
                "iteration": iteration,
                "tool": tool_name,
                "args": arguments,
                "observation": result
            })

        if pending_tools:
            answer = "Đã vượt quá số bước xử lý tối đa; một số yêu cầu chưa được hoàn tất."
            self.trace.append({"step": "max_iterations_reached"})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "max_iterations_reached"
            }

        answer_parts = []
        catalog_result = observations.get("search_product_catalog")
        tool_error = False
        if catalog_result is not None:
            if catalog_result and "error" in catalog_result[0]:
                answer_parts.append(
                    f"Không thể tra cứu danh mục: {catalog_result[0]['error']}"
                )
                tool_error = True
            elif catalog_result:
                answer_parts.append(
                    "Các lựa chọn phù hợp:\n" + "\n".join(
                        f"- {product['name']} ({product['price_vnd']:,} VNĐ)"
                        for product in catalog_result
                    )
                )
            else:
                answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")

        ticket_result = observations.get("submit_support_ticket")
        if ticket_result is not None:
            answer_parts.append(
                f"Đã ghi nhận yêu cầu hỗ trợ cho {ticket_result['customer_name']}. "
                f"Mã ticket: {ticket_result['ticket_id']}."
            )

        if not answer_parts:
            if is_faq:
                answer_parts.append(
                    "Theo thông tin hiện có trong danh mục, pin VinFast VF 5 Plus "
                    "được bảo hành 10 năm. Thời hạn này được nêu cho mẫu VF 5 Plus."
                )
            else:
                answer_parts.append(
                    "Tôi có thể hỗ trợ về sản phẩm, dịch vụ Vingroup và yêu cầu hỗ trợ khách hàng."
                )

        answer = "\n".join(answer_parts)
        self.trace.append({"step": "final_answer", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": max(iteration, 1),
            "status": "tool_error" if tool_error else "completed"
        }


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _parse_catalog_args(normalized_text: str) -> Dict[str, Any]:
    category = "du_lich" if any(
        term in normalized_text for term in ("vinpearl", "du lich", "resort", "khach san")
    ) else "xe_dien"
    max_price = 999999999999

    price_with_unit = re.search(
        r"(\d[\d.,]*)\s*(ty|trieu|nghin|ngan)\b",
        normalized_text
    )
    if price_with_unit:
        amount = _parse_number(price_with_unit.group(1))
        multiplier = {
            "ty": 1_000_000_000,
            "trieu": 1_000_000,
            "nghin": 1_000,
            "ngan": 1_000
        }[price_with_unit.group(2)]
        max_price = int(amount * multiplier)
    else:
        price_with_currency = re.search(
            r"(?:duoi|toi da|gia)\s*(\d[\d.,]*)\s*(?:vnd|dong)\b",
            normalized_text
        )
        if price_with_currency:
            max_price = int(_parse_number(price_with_currency.group(1)))
        else:
            plain_price = re.search(r"(?:duoi|toi da)\s*(\d{7,})\b", normalized_text)
            if plain_price:
                max_price = int(plain_price.group(1))

    return {"category": category, "max_price": max_price}


def _parse_number(value: str) -> float:
    if "." in value and "," in value:
        value = value.replace(".", "").replace(",", "")
    elif "." in value or "," in value:
        separator = "." if "." in value else ","
        parts = value.split(separator)
        if len(parts[-1]) == 3:
            value = "".join(parts)
        else:
            value = value.replace(",", ".")
    return float(value)


def _parse_ticket_args(user_input: str, normalized_text: str) -> Dict[str, str]:
    name_match = re.search(
        r"(?:tôi\s+tên|tên\s+tôi\s+là|tên\s+là|mình\s+tên\s+là)\s+([^,.;!?]+)",
        user_input,
        flags=re.IGNORECASE
    )
    customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

    if name_match:
        issue = user_input[name_match.end():]
        issue = re.sub(r"^\s*[,;:\-–.]\s*", "", issue)
    else:
        issue = re.sub(
            r"^.*?(?:ghi nhận phản hồi|ghi nhận yêu cầu|báo lỗi|gặp vấn đề)\s*[:,-]?\s*",
            "",
            user_input,
            flags=re.IGNORECASE
        )

    issue = re.split(r"(?<=[.!?])\s+", issue, maxsplit=1)[0]
    issue = re.split(
        r",\s*(?:mức độ|độ ưu tiên|priority)\b",
        issue,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]
    issue = re.sub(r"\bcủa tôi\b", "", issue, flags=re.IGNORECASE)
    issue = re.sub(r"\btôi đặt\b", "đặt", issue, flags=re.IGNORECASE)
    issue = re.sub(r"\s+", " ", issue).strip(" ,;:-.!?")
    if issue:
        issue = issue[0].upper() + issue[1:]
    else:
        issue = user_input.strip()

    priority = "medium"
    if any(term in normalized_text for term in ("nghiem trong", "khan cap", "xu ly gap", "rat gap")):
        priority = "high"
    elif any(term in normalized_text for term in ("muc do thap", "uu tien thap", "khong gap")):
        priority = "low"

    return {
        "customer_name": customer_name,
        "issue_description": issue,
        "priority": priority
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
