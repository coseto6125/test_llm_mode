# langgraph_flow.py

import asyncio
import os
import platform
from typing import Any, Literal, NotRequired, TypedDict

# 使用try-except避免Windows系統錯誤
try:
    import uvloop

    has_uvloop = True
except ImportError:
    has_uvloop = False

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

# 假設 raccoon_client.py 與此文件在同一目錄或 PYTHONPATH 中
# 如果不在，請調整導入路徑
from raccoon_client import RaccoonAIClient

# --- 配置與初始化 ---

# 配置日誌
logger.add("langgraph_flow.log", rotation="10 MB", retention="7 days", level="DEBUG")

# 加載環境變數 (例如 OPENAI_API_KEY)
load_dotenv()

# 檢查 OpenAI API Key 是否設置
if not os.getenv("OPENAI_API_KEY_QA"):
    logger.warning("OPENAI_API_KEY_QA 環境變數未設置，OpenAI LLM 可能無法工作。")
    raise ValueError("OPENAI_API_KEY environment variable not set.")  # 或者直接報錯


# --- Langgraph 狀態定義 ---


class FlowState(TypedDict):
    source_input: NotRequired[str]
    llm2_output: NotRequired[str]
    llm3_output: NotRequired[str]
    merged_output: NotRequired[str]
    final_output: NotRequired[str]


# --- 自訂 Runnable 封裝 RaccoonAIClient ---


# 現在支援串流模式
class RaccoonRunnable(BaseModel):
    """封裝 RaccoonAIClient 的 Langchain Runnable"""

    client: RaccoonAIClient = Field(default_factory=RaccoonAIClient)
    brand_id: str = "137"  # 預設 brand_id
    stream: bool = False  # 是否使用串流模式

    class Config:
        arbitrary_types_allowed = True

    async def _invoke_raccoon(self, message: str, config: RunnableConfig | None = None) -> dict[str, Any]:
        """異步調用 Raccoon AI Client"""
        # 確保在異步上下文中使用 client
        async with self.client as client:
            try:
                logger.debug(f"調用 Raccoon AI (Brand ID: {self.brand_id})，訊息: {message}...")

                if self.stream:
                    # 使用串流模式
                    chunks = []
                    # 添加測試信息，確認有開始串流
                    logger.debug(f"開始串流處理 (Brand ID: {self.brand_id})")
                    print(f"\n[開始處理 Agent {self.brand_id} 的回應]", flush=True)

                    async for chunk in client.stream_chat(message=message, brand_id=self.brand_id, reset_history=True):
                        # 即時輸出
                        print(chunk, end="", flush=True)
                        chunks.append(chunk)

                    # 構建相容於非串流格式的回應
                    full_content = "".join(chunks)
                    logger.debug(
                        f"Raccoon AI 串流回應完成 (Brand ID: {self.brand_id}), 收集到內容長度: {len(full_content)}"
                    )

                    if not full_content.strip():
                        logger.warning(f"Agent {self.brand_id} 回應為空，使用預設訊息")
                        full_content = f"Agent {self.brand_id} 未提供有效回應。"

                    print(f"\n[Agent {self.brand_id} 回應完成]\n", flush=True)

                    # 嘗試解析 JSON 結構，獲取 quick_replies.text 欄位
                    try:
                        # 檢查是否為二進制字串格式，通常以 b' 開頭
                        if full_content.startswith("b'"):
                            # 處理二進制字串，去除 b' 和結尾的 ' 並解碼
                            import ast
                            import json

                            # 使用 ast.literal_eval 安全地解析二進制字串為 Python 字節對象
                            binary_str = ast.literal_eval(full_content)

                            # 解碼為字符串，處理可能的 UTF-8 錯誤
                            decoded_content = binary_str.decode("utf-8", errors="replace")

                            # 解析 JSON
                            json_data = json.loads(decoded_content)

                            # 檢查並提取 quick_replies.text
                            if isinstance(json_data, dict) and "quick_replies" in json_data:
                                if (
                                    isinstance(json_data["quick_replies"], dict)
                                    and "text" in json_data["quick_replies"]
                                ):
                                    text_content = json_data["quick_replies"]["text"]
                                    logger.info(f"成功從 JSON 提取 quick_replies.text: {text_content[:100]}...")
                                    full_content = text_content

                    except Exception as e:
                        logger.warning(f"解析 JSON 結構時發生錯誤: {str(e)}，將使用原始內容")

                    # 構建與非串流回應格式相容的結構
                    return {"response": [{"content": full_content}]}
                else:
                    # 原有的非串流模式
                    response = await client.chat(message=message, brand_id=self.brand_id, reset_history=True)
                    logger.debug(f"Raccoon AI 回應 (Brand ID: {self.brand_id}): {str(response)[:100]}...")
                    return response  # 直接返回完整回應字典

            except Exception:
                logger.exception(f"調用 Raccoon AI (Brand ID: {self.brand_id}) 時發生錯誤")
                raise  # 重新拋出，讓 Langgraph 的錯誤處理機制接管

    def as_runnable(self) -> RunnableLambda:
        """將此類轉換為 RunnableLambda"""
        return RunnableLambda(self._invoke_raccoon)


# --- Langgraph 節點定義 ---


async def llm_agent_node(state: FlowState, agent_id: int, raccoon_runnable: RaccoonRunnable) -> dict[str, Any]:
    """執行 LLM Agent (Raccoon) 的節點"""
    logger.info(f"進入 LLM Agent {agent_id} 節點")
    message = state["source_input"]
    if not message:
        logger.warning(f"LLM Agent {agent_id}: 原始訊息為空")
        return {f"llm_{agent_id}_output": {"error": "Original message is empty."}}

    try:
        # 串流模式會在 _invoke_raccoon 中即時輸出
        response = await raccoon_runnable.as_runnable().ainvoke(message)
        logger.info(f"LLM Agent {agent_id} 執行完成")
        return {f"llm_{agent_id}_output": response}
    except Exception as e:
        logger.exception(f"LLM Agent {agent_id} 執行時發生錯誤")
        return {f"llm_{agent_id}_output": {"error": f"Agent {agent_id} failed: {str(e)}"}}


async def aggregator_node(state: FlowState) -> dict[str, Any]:
    """聚合 LLM Agent 2 和 3 結果的節點"""
    logger.info("進入 Aggregator 節點")
    llm_2_resp = state.get("llm2_output")
    llm_3_resp = state.get("llm3_output")

    # 檢查是否有錯誤
    error_2 = llm_2_resp.get("error") if isinstance(llm_2_resp, dict) else None
    error_3 = llm_3_resp.get("error") if isinstance(llm_3_resp, dict) else None

    if error_2 or error_3:
        errors = [e for e in [error_2, error_3] if e]
        error_msg = f"Aggregation failed due to upstream errors: {'; '.join(errors)}"
        logger.error(error_msg)
        return {"error": error_msg}  # 將錯誤記錄到狀態中，終止流程或導向錯誤處理分支

    # 提取內容 - 根據不同的回應結構提取文本內容
    content_2 = ""
    # 處理串流模式返回的格式
    if isinstance(llm_2_resp, dict) and isinstance(llm_2_resp.get("response"), list) and llm_2_resp["response"]:
        response_item = llm_2_resp["response"][0]
        if isinstance(response_item, dict):
            # 直接獲取內容 (串流模式下應該是直接字串)
            if isinstance(response_item.get("content"), str):
                content_2 = response_item.get("content", "")
            # 非串流模式下的複雜結構
            elif isinstance(response_item.get("content"), list) and response_item["content"]:
                content_2 = response_item["content"][0].get("text", "")

    content_3 = ""
    # 處理串流模式返回的格式
    if isinstance(llm_3_resp, dict) and isinstance(llm_3_resp.get("response"), list) and llm_3_resp["response"]:
        response_item = llm_3_resp["response"][0]
        if isinstance(response_item, dict):
            # 直接獲取內容 (串流模式下應該是直接字串)
            if isinstance(response_item.get("content"), str):
                content_3 = response_item.get("content", "")
            # 非串流模式下的複雜結構
            elif isinstance(response_item.get("content"), list) and response_item["content"]:
                content_3 = response_item["content"][0].get("text", "")

    # 檢查是否有內容
    if not content_2.strip() and not content_3.strip():
        logger.warning("兩個 Agent 的回應均為空")
        merged_output = "兩個 Agent 均未返回有效回應。請檢查連接或嘗試不同的查詢。"
    else:
        # 簡單聚合：將兩個回應拼接起來
        merged_output = f"來自 Agent 2 的觀點:\n{content_2}\n\n來自 Agent 3 的觀點:\n{content_3}"

    logger.info("Aggregator 節點執行完成")
    logger.debug(f"聚合後內容: {merged_output[:100]}...")

    return {"merged_output": merged_output}


async def final_llm_node(state: FlowState) -> dict[Literal["final_output"], str]:
    """執行最終 LLM Agent (OpenAI) 並串流輸出的節點"""
    logger.info("進入 Final LLM 節點")
    merged_output = state.get("merged_output")

    if state.get("error"):  # 檢查上游是否有錯誤
        logger.warning(f"由於上游錯誤，跳過 Final LLM 節點: {state['error']}")
        # 可以選擇直接結束或返回錯誤狀態
        # return {"final_output": None, "error": state["error"]}
        # 這裡選擇讓流程自然結束，錯誤已記錄
        return {"final_output": "流程執行中發生錯誤"}

    if not merged_output:
        logger.warning("Final LLM: 聚合內容為空")
        return {"final_output": "聚合內容為空，無法生成最終回應。"}

    # 記錄接收到的聚合內容，診斷問題
    logger.debug(f"Final LLM 收到的聚合內容: {merged_output[:150]}...")

    # 檢查聚合內容是否只有標題而沒有實質內容
    if merged_output.strip() == "來自 Agent 2 的觀點:\n\n\n來自 Agent 3 的觀點:\n" or len(merged_output.strip()) < 50:
        logger.warning("聚合內容缺少實質內容，使用預設訊息替代")
        merged_output = """來自 Agent 2 的觀點:
這是一個測試對話，我會根據評估標準給您提供客服對話的專業評分和建議。

來自 Agent 3 的觀點:
聚合內容似乎缺少實質內容，可能是由於連接問題或API回應異常。建議檢查API連接和stream_chat方法的實現。
"""

    # 從 prompt 導入系統提示
    from prompt import basic_evaluator_prompt

    # 初始化 OpenAI LLM (串流模式)
    # 可以根據需要調整模型名稱和參數
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, streaming=True)

    # 建立系統提示和用戶提示
    system_message = SystemMessage(content=basic_evaluator_prompt)
    human_message = HumanMessage(content=merged_output)

    # 異步串流調用
    try:
        print("\n--- 評估結果 ---\n", flush=True)
        async for chunk in llm.astream([system_message, human_message]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\n", flush=True)
        logger.info("Final LLM 串流輸出完成")
        return {"final_output": "done"}

    except Exception as e:
        logger.exception("Final LLM 執行時發生錯誤")
        return {"final_output": f"Final LLM failed: {str(e)}"}


# --- Langgraph 圖構建 ---


def build_graph() -> StateGraph:
    """構建 Langgraph 流程圖"""
    workflow = StateGraph(FlowState)

    # 使用串流模式
    raccoon_agent_2 = ChatOpenAI(model="gpt-4-1106-nano", temperature=0.7, streaming=True)
    raccoon_agent_3 = ChatOpenAI(model="gpt-4-1106-nano", temperature=0.7, streaming=True)

    # 偏函數應用，將 agent_id 和 runnable 綁定到節點函數
    # 使用 functools.partial 替代 lambda 以提高效能 (雖然此處差異不大)
    from functools import partial

    llm_agent_2_node_partial = partial(llm_agent_node, agent_id=137, raccoon_runnable=raccoon_agent_2.as_runnable())
    llm_agent_3_node_partial = partial(llm_agent_node, agent_id=137, raccoon_runnable=raccoon_agent_3.as_runnable())

    workflow.add_node("llm_agent_2", llm_agent_2_node_partial)
    workflow.add_node("llm_agent_3", llm_agent_3_node_partial)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("final_llm", final_llm_node)

    # 設置一個虛擬的起始分支節點，確保並行執行
    def start_branch(state: FlowState):
        # 這個節點不做任何事，只是為了觸發後續的並行分支
        logger.info("流程開始，準備分支到 LLM Agent 2 和 3")
        return {}  # 返回空字典，不修改狀態

    workflow.add_node("start_branch", start_branch)
    workflow.set_entry_point("start_branch")
    workflow.add_edge("start_branch", "llm_agent_2")
    workflow.add_edge("start_branch", "llm_agent_3")

    # 添加聚合和最終節點的邊
    workflow.add_edge("llm_agent_3", "aggregator")  # Agent 3 完成後也到 aggregator
    workflow.add_edge("aggregator", "final_llm")

    # 添加結束點
    workflow.add_edge("final_llm", END)

    # 編譯圖
    app = workflow.compile()
    logger.info("Langgraph 圖構建完成")
    return app


# --- 主執行函數 ---


async def main(initial_message: str):
    """主執行函數，運行 Langgraph 流程並處理串流輸出"""
    app = build_graph()

    # 初始狀態
    initial_state: FlowState = {"source_input": initial_message}

    logger.info(f"開始執行 Langgraph 流程，初始訊息: {initial_message}...")

    # 使用 astream_events 來獲取更詳細的事件流，包括中間狀態和節點輸出
    final_state = None
    try:
        # 在 Windows 上使用 ainvoke 而非 astream_events，避免 aiodns 錯誤
        if platform.system() == "Windows":
            logger.info("Windows 系統下使用 ainvoke 模式")
            final_state = await app.ainvoke(initial_state)
            logger.info("Langgraph 流程執行完畢")
        else:
            # 非 Windows 系統使用 astream_events 獲取詳細事件
            async for event in app.astream_events(initial_state, version="v1"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    # 這個事件類型適用於 Langchain LLM 的串流塊
                    content = event["data"]["chunk"].content
                    if content:
                        print(content, end="", flush=True)  # 即時打印串流內容
                elif kind == "on_graph_end":
                    # 圖執行結束，獲取最終狀態
                    final_state = event["data"]["output"]
                    logger.info("Langgraph 流程執行完畢")
                    break  # 圖結束後退出循環

        print("\n--- 流程結束 ---")

        if final_state and isinstance(final_state, dict):
            logger.debug(f"最終狀態: { {k: v for k, v in final_state.items() if k != 'final_output'} }")
            if error := final_state.get("error"):
                logger.error(f"流程執行中發生錯誤: {error}")
                print(f"\n錯誤: {error}")
        else:
            logger.error("未能獲取最終狀態")

    except Exception as e:
        logger.exception(f"執行流程時發生錯誤: {e}")
        print(f"\n執行過程中發生錯誤: {e}")


if __name__ == "__main__":
    # 為 Windows 系統設置合適的事件循環政策
    if platform.system() == "Windows":
        logger.info("在 Windows 上設置 WindowsSelectorEventLoopPolicy")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # 在非 Windows 系統上使用 uvloop (如果可用)
    elif has_uvloop:
        try:
            uvloop.install()
            logger.info("使用 uvloop 加速異步操作")
        except Exception as e:
            logger.warning(f"無法安裝 uvloop: {e}")
    else:
        logger.info("使用標準事件循環")

    # 測試訊息
    from prompt import api_request_data

    # 運行主函數
    asyncio.run(main(api_request_data))
